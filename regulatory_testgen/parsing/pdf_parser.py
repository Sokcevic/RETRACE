"""Parse a regulation into Clause objects with structure taken from the PDF.

The clause hierarchy (which clauses exist, their ids and nesting) comes from
structure_decoder, which reads the PDF layout and numbering. Clause *content*
comes from MinerU's Markdown when it is available, because downstream stages
depend on it: table extraction reads MinerU's HTML tables and the chat UI maps
clause text back to MinerU's page blocks. Each decoded heading is aligned to its
Markdown line (forward-only cursor, matched on heading text), and a clause's text
is the Markdown between its heading and the next aligned heading. Without
Markdown, or for a heading that cannot be aligned, the text comes from the PDF
lines instead.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from regulatory_testgen.data_models.core import Clause
from regulatory_testgen.parsing.markdown_parser import (
    DOT_LEADER_RE,
    HTML_TABLE_END_RE,
    HTML_TABLE_START_RE,
    MARKDOWN_TABLE_ROW_RE,
    ClauseBuilder,
    deduplicate_clauses,
    looks_like_title,
    looks_like_toc_entry,
)
from regulatory_testgen.parsing.structure_decoder import (
    DEFAULT_START_RE,
    Heading,
    Line,
    decode_structure,
    pdf_layout_lines,
)
from regulatory_testgen.parsing.table_merge import merge_multipage_tables
from regulatory_testgen.structure_ids import parent_of

logger = logging.getLogger(__name__)

_ALIGN_WINDOW = 400  # Markdown lines searched ahead of the cursor for each heading
_MIN_KEY = 6  # shorter normalized keys match too much
_LEADING_NUMBER_RE = re.compile(
    r"^#*\s*(?:Annex\s+\d+(?:\s*[-–]\s*Appendix(?:\s+\d+)?)?|APP\s*\d+|[A-Z]?\d+(?:\.\d+)*\.?|\(\d+\)"
    r"|[IVX]+\.|[A-Z]\.)\s*",
    re.IGNORECASE,
)


def parse_pdf_clauses(
    pdf_path: str | Path,
    markdown_path: str | Path | None = None,
    *,
    start: str = DEFAULT_START_RE,
    profile: dict | None = None,
) -> list[Clause]:
    """Clauses with PDF-decoded structure; content from MinerU Markdown if given.

    profile, if given, receives the decoder's document-level decisions plus
    alignment statistics ("aligned", "unaligned")."""
    profile = profile if profile is not None else {}
    lines = pdf_layout_lines(pdf_path)
    headings = sorted(decode_structure(lines, start, profile=profile), key=lambda h: h.line_index)
    if not headings:
        logger.warning("No structure decoded from %s", pdf_path)
        return []

    md_lines: list[str] = []
    anchors: list[int | None] = [None] * len(headings)
    if markdown_path is not None:
        text, _ = merge_multipage_tables(Path(markdown_path).read_text(encoding="utf-8"))
        md_lines = text.splitlines()
        anchors = _align(headings, md_lines)
    titles = {}
    for h, anchor in zip(headings, anchors):
        title = _title_for(h, lines)
        if title and anchor is not None and h.kind not in ("container", "note", "virtual"):
            rest = _norm(_LEADING_NUMBER_RE.sub("", md_lines[anchor].strip(), count=1))
            if len(rest) > len(_norm(title)) + 10:
                title = ""  # first visual line of a wrapped paragraph, not a title
        titles.setdefault(h.clause_id, title)
    profile["aligned"] = sum(a is not None for a in anchors)
    profile["unaligned"] = [h.clause_id for h, a in zip(headings, anchors) if a is None and md_lines]
    if md_lines and profile["unaligned"]:
        logger.info("%d/%d headings not found in %s; using PDF text for them",
                    len(profile["unaligned"]), len(headings), markdown_path)

    clauses: list[Clause] = []
    md_source = Path(markdown_path).as_posix() if markdown_path is not None else None
    pdf_source = Path(pdf_path).as_posix()

    # front matter: everything before the first heading
    first_md = next((a for a in anchors if a is not None), None)
    if md_lines and first_md:
        clauses.append(_build("front-matter", "Front matter", [], md_source,
                              list(enumerate(md_lines[:first_md], start=1)), pseudo=True))
    elif not md_lines and headings[0].line_index > 0:
        clauses.append(_build("front-matter", "Front matter", [], pdf_source,
                              [(i + 1, ln.text) for i, ln in enumerate(lines[: headings[0].line_index])],
                              pseudo=True))

    aligned_positions = [a for a in anchors if a is not None]
    for k, (h, anchor) in enumerate(zip(headings, anchors)):
        path = _section_path(h.clause_id, titles)
        if anchor is not None:
            nxt = next((a for a in aligned_positions if a > anchor), len(md_lines))
            body = list(enumerate(md_lines[anchor:nxt], start=anchor + 1))
            body[0] = (body[0][0], _strip_heading(body[0][1], h, titles[h.clause_id]))
            if h.kind in ("container", "virtual") and titles[h.clause_id]:
                # MinerU repeats a multi-line title as separate heading lines
                t = _norm(titles[h.clause_id])
                body = [(n, "" if _norm(s) and _norm(s) in t else s) for n, s in body[:6]] + body[6:]
            clauses.append(_build(h.clause_id, titles[h.clause_id], path, md_source, body))
        else:
            end = headings[k + 1].line_index if k + 1 < len(headings) else len(lines)
            rows = [(h.line_index + 1, "" if titles[h.clause_id] else h.text)]
            rows += [(i + 1, lines[i].text) for i in range(h.line_index + 1, end)]
            if h.kind in ("container", "virtual") and titles[h.clause_id]:
                t = _norm(titles[h.clause_id])  # title lines are already the clause title
                rows = rows[:1] + [(n, "" if k < 4 and _norm(s) and _norm(s) in t else s)
                                   for k, (n, s) in enumerate(rows[1:])]
            clauses.append(_build(h.clause_id, titles[h.clause_id], path, pdf_source, rows))
    return deduplicate_clauses(clauses)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _title_for(h: Heading, lines: list[Line]) -> str:
    if h.kind in ("container", "virtual"):
        # an annex title / scenario subtitle: the bold lines that follow, joined
        first = lines[h.line_index]
        parts = [] if h.kind == "container" else [h.text or h.label]
        for ln in lines[h.line_index + 1: h.line_index + 5]:
            if not ln.bold or re.match(r"^\d", ln.text) or (parts and h.kind == "container"
                                                            and abs(ln.size - first.size) > 0.5):
                break
            parts.append(ln.text)
            if h.kind == "virtual":
                break
        return " ".join(parts)
    if h.kind == "note":
        return h.text or h.label
    if not h.text or h.text[0] in "\"“'‘":
        return ""  # quoted definitions are paragraphs, whatever their first line says
    return h.text if looks_like_title(h.text) else ""


def _section_path(clause_id: str, titles: dict[str, str]) -> list[str]:
    path, p = [], parent_of(clause_id)
    while p:
        path.append(f"{p} {titles.get(p, '')}".strip())
        p = parent_of(p)
    return path[::-1]


def _build(clause_id, title, section_path, source, rows, pseudo=False) -> Clause:
    builder = ClauseBuilder(clause_id=clause_id, title=title, source=source or "",
                            line_start=rows[0][0] if rows else 0, section_path=section_path,
                            is_pseudo_clause=pseudo)
    for line_no, text in rows:
        builder.add_line(text, line_no)
    return builder.to_clause()


def _norm(text: str) -> str:
    # MinerU writes symbols as LaTeX ("$( P B C )$", "$0 ^ { \\circ }$"): keep the
    # content, drop the markup
    text = re.sub(r"<[^>]+>|\\[a-zA-Z]+", " ", text)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _keys(h: Heading) -> list[str]:
    if h.kind == "container":
        return [_norm(h.label)]
    if h.kind in ("note", "virtual"):
        return [_norm(h.text or h.label)]
    tail = _norm(h.text)[:40]
    with_number = _norm(f"{h.label} {h.text}")[:40]
    # last resort: a shorter prefix survives glued footnote markers and LaTeX
    # symbols further into the line ("nominal9 peak", "0 °C")
    short = tail[:24] if len(tail) >= 20 else ""
    keys = [k for k in (tail, with_number, short) if len(k) >= _MIN_KEY]
    return keys or [with_number]


def _alignable(md_lines: list[str]) -> list[bool]:
    """Markdown lines a heading may anchor to: not inside tables, not TOC entries."""
    ok, in_table = [], False
    for line in md_lines:
        s = line.strip()
        if HTML_TABLE_START_RE.search(s):
            in_table = not HTML_TABLE_END_RE.search(s)
            ok.append(False)
            continue
        if in_table:
            in_table = not HTML_TABLE_END_RE.search(s)
            ok.append(False)
            continue
        ok.append(bool(s) and not MARKDOWN_TABLE_ROW_RE.match(s) and not DOT_LEADER_RE.search(s)
                  and not looks_like_toc_entry(s))
    return ok


def _align(headings: list[Heading], md_lines: list[str]) -> list[int | None]:
    ok = _alignable(md_lines)
    norms = [_norm(line) if good else "" for line, good in zip(md_lines, ok)]
    anchors: list[int | None] = []
    cursor = _body_start(md_lines)
    for h in headings:
        # an inline sub-rule ("1.1.1 (1) The Rules ...") shares its rule's line
        lo = max(cursor - 1, 0) if h.kind == "subrule" else cursor
        found = None
        for key in _keys(h):
            for j in range(lo, min(len(md_lines), lo + _ALIGN_WINDOW)):
                if norms[j] and key in norms[j]:
                    found = j
                    break
            if found is not None:
                break
        if found is not None and h.kind in ("clause", "subheading"):
            prev = found - 1
            while prev >= lo and not md_lines[prev].strip():
                prev -= 1
            if prev >= lo and re.fullmatch(r"#*\s*" + re.escape(h.label) + r"\.?", md_lines[prev].strip()):
                found = prev  # number split onto its own line ("# 2.15." then the text)
        anchors.append(found)
        if found is not None:
            cursor = found + 1
    return anchors


def _toc_like(line: str) -> bool:
    s = line.strip()
    return bool(s) and (bool(DOT_LEADER_RE.search(s)) or looks_like_toc_entry(s)
                        or (len(s) < 100 and bool(re.match(r"^#*\s*(\d+|Annex|Appendix|APP)\b", s))
                            and bool(re.search(r"\.\s*\d+\s*$", s))))


def _body_start(md_lines: list[str]) -> int:
    """Skip a leading table of contents: start after the last run of three or more
    consecutive TOC-like lines in the first third of the document."""
    last_toc, run = -1, []
    for j, line in enumerate(md_lines[: max(1, len(md_lines) // 3)]):
        if not line.strip():
            continue
        if _toc_like(line):
            run.append(j)
            if len(run) >= 3:
                last_toc = j
        else:
            run = []
    return last_toc + 1


def _strip_heading(line: str, h: Heading, title: str) -> str:
    """The first Markdown line of a clause minus its number and title."""
    rest = _LEADING_NUMBER_RE.sub("", line.strip(), count=1) if h.kind != "virtual" else line.strip()
    if title and _norm(rest).startswith(_norm(title)):
        return ""
    return rest
