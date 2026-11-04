"""Recover a document's clause hierarchy from PDF layout + numbering.

MinerU flattens every heading to one Markdown level and drops or splits clause
numbers, so the Markdown alone cannot tell us the document tree. This module
reads the PDF directly and treats the numbering as a grammar:

  * every line that starts with a number ("5.2.1", "A1.2", "II.", "B.") is a
    candidate heading;
  * a candidate is only accepted as part of a chain of legal successors
    (5.2.1 -> 5.2.2 / 5.2.1.1 / 5.3 / 6), and the best chain is chosen globally
    (longest path in a DAG), so footnote markers, table rows and figure labels
    cannot derail the tree the way a greedy left-to-right scan does;
  * small, document-level conventions are profiled rather than assumed: where
    arabic numbering restarts (per Roman part, per letter section, never),
    whether "(n)" lines are sub-rules or inline lists, and which recurring bold
    label introduces note blocks ("Guidance").

Containers ("Annex 3", "Annex 3 - Appendix 1", "APP 2") reset the numbering and
namespace the ids exactly like regulatory_testgen.structure_ids does
("Annex 3 / 5.1"), so the output plugs into the existing graph unchanged.

Layout comes from PyMuPDF. If the PDF is tagged and pypdf is installed, text
inside tagged tables is ignored (cross-reference tables otherwise look like
clause lines).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?(?=\s|$|[A-Za-z\"“(])")
ALNUM_RE = re.compile(r"^([A-Z]\d+(?:\.\d+)+)\.?(?=\s)")  # letter-prefixed "A1.1.2"
PAREN_RE = re.compile(r"^\((\d+)\)\s+(?=\S)")  # parenthesized sub-rule "(2) ..."
TOC_PAGE_RE = re.compile(r"(?:(\.\s*){4,}|…{2,})\s*\d+\s*$")
TOC_RE = re.compile(r"(\.\s*){4,}\S*\s*\d*\s*$|…{2,}")  # dot leaders
WORD_RE = re.compile(r"[^\W\d_]{2,}")
INVISIBLE_RE = re.compile("[​-‏‪-‮⁠﻿]")
ANNEX_RE = re.compile(
    r"^(Annex\s+\d+(?:\s*[-–]\s*Appendix(?:\s+\d+)?)?|Appendix\s+\d+)\s*$", re.IGNORECASE
)
APP_RE = re.compile(r"^(APP)\s*(\d+)\b")  # ADGM appendix heading, title on the same line
# Keyword-led numbered heading: "PART 7", "CHAPTER 7.2" (ADN, ADR); the number joins the arabic chain
# Volume title page of a multi-volume PDF (ADN 2025: Parts 1, 4-9 in Volume I,
# Parts 2-3 in Volume II), so the numbering may restart anywhere at the boundary.
VOLUME_RE = re.compile(r"^Volume\s+([IVX]+|\d+)\s*$", re.IGNORECASE)
# Range heading "7.2.4.30 to 7.2.4.39 (Reserved)": the chain continues after the last number
RANGE_RE = re.compile(r"^(?:to|and|-|–)\s+(\d+(?:\.\d+)+)\b")
KEYWORD_NUM_RE = re.compile(r"^(?:PART|Part|CHAPTER|Chapter)\s+(\d+(?:\.\d+)*)\.?(?=\s|$|:)")
ROMAN_RE = re.compile(r"^([IVX]+|[Ⅰ-Ⅻ])\.\s*(?=[A-Z“\"])")
LETTER_RE = re.compile(r"^([A-Z])\.\s*(?=[A-Z“\"][a-z]|[A-Z]\b)")
ROMAN_UNICODE = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ"
DEFAULT_START_RE = r"^(1\.?|I\.|Ⅰ\.)\s+\S"

MAX_GAP = 3
RESTART_COST = 1.0  # score a restart scope pays per extra numbering restart
RESYNC_COST = 4.0  # score a chain pays to jump forward over a broken successor link


@dataclass(frozen=True)
class Line:
    """One visual line of the PDF (spans sharing a baseline)."""

    page: int  # 1-based
    text: str
    size: float  # largest font size on the line, in points
    bold: bool
    top: float  # points from the top of the page
    x0: float


@dataclass(frozen=True)
class Heading:
    """A structural node recovered by the decoder."""

    clause_id: str  # namespaced id, e.g. "5.2.1", "Annex 3 / 2.1", "II / 5.1"
    label: str  # the number as printed ("5.2.1", "A1.1", "(2)", "Guidance")
    line_index: int  # index into the Line list where the heading starts
    text: str  # rest of the heading line after the number
    kind: str  # clause | container | roman | letter | subheading | subrule | note | virtual


# ---------------------------------------------------------------------------
# Layout extraction
# ---------------------------------------------------------------------------


def pdf_layout_lines(pdf_path: str | Path, skip_tables: bool = True) -> list[Line]:
    """Visual lines of a PDF with font size and weight, page furniture bands removed."""
    try:
        import pymupdf as fitz
    except ImportError:  # PyMuPDF < 1.24.3 only ships the "fitz" name
        import fitz

    tables = table_regions(pdf_path) if skip_tables else {}
    lines: list[Line] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_index, page in enumerate(doc):
            height, width = page.rect.height, page.rect.width
            boxes = tables.get(page_index, [])
            spans_on_page: list[tuple[float, tuple[float, float, str, float, bool]]] = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        text = span["text"]
                        if not text.strip():
                            continue
                        x0, y0, x1, y1 = span["bbox"]
                        if y0 < 0.05 * height or y1 > 0.95 * height:
                            continue  # running header / page number band
                        cx, cy = (x0 + x1) / 2, height - (y0 + y1) / 2
                        if any(bx0 <= cx <= bx1 and by0 <= cy <= by1 for bx0, by0, bx1, by1 in boxes):
                            continue  # inside a tagged table
                        bold = bool(span["flags"] & 16) or "bold" in span["font"].lower()
                        spans_on_page.append((span["origin"][1], (x0, y0, text, span["size"], bold)))
            # Group spans into visual lines by baseline. Clustering with a
            # tolerance instead of fixed buckets keeps a margin number and its
            # text together when their baselines differ by a fraction of a
            # point but straddle a bucket boundary (ADN: "7.2.4.15.3" at 192.6,
            # its text at 193.1).
            rows: list[list[tuple[float, float, str, float, bool]]] = []
            last_baseline = None
            for baseline, span in sorted(spans_on_page, key=lambda s: s[0]):
                if last_baseline is None or baseline - last_baseline > 3.5:
                    rows.append([])
                rows[-1].append(span)
                last_baseline = baseline
            for row in rows:
                spans = sorted(row)
                text = INVISIBLE_RE.sub("", " ".join(" ".join(s[2].split()) for s in spans)).strip()
                if not text:
                    continue
                lines.append(Line(
                    page=page_index + 1,
                    text=text,
                    size=round(max(s[3] for s in spans), 1),
                    bold=any(s[4] for s in spans if s[2].strip()),
                    top=min(s[1] for s in spans),
                    x0=spans[0][0],
                ))
    return lines


def table_regions(pdf_path: str | Path) -> dict[int, list[tuple[float, float, float, float]]]:
    """Boxes (PDF points, origin bottom-left) of tagged /Table elements, per 0-based page.

    Walks the structure tree, collects the marked-content ids under each table and
    replays the page content streams to find where that content is drawn. Untagged
    PDFs, or a missing pypdf, yield no regions.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.debug("pypdf not installed; table filtering disabled")
        return {}
    reader = PdfReader(str(pdf_path))
    st = reader.trailer["/Root"].get("/StructTreeRoot")
    if st is None:
        return {}
    st = st.get_object()
    rolemap = {str(k): str(v) for k, v in (st.get("/RoleMap") or {}).items()}
    owned: dict[tuple[int, int], int] = {}
    counter = [0]

    def walk(node, page, owner):
        if isinstance(node, int):
            if owner is not None and page is not None:
                owned[(page, node)] = owner
            return
        node = node.get_object()
        if isinstance(node, list):
            for kid in node:
                walk(kid, page, owner)
            return
        if not hasattr(node, "get"):
            return
        pg = node.get("/Pg")
        if pg is not None and hasattr(pg, "idnum"):
            page = pg.idnum
        if str(node.get("/Type", "")) == "/MCR":
            if owner is not None and page is not None:
                owned[(page, int(node["/MCID"]))] = owner
            return
        tag = node.get("/S")
        if tag is not None and owner is None and rolemap.get(str(tag), str(tag)) == "/Table":
            counter[0] += 1
            owner = counter[0]
        if node.get("/K") is not None:
            walk(node["/K"], page, owner)

    walk(st.get("/K"), None, None)
    result: dict[int, list[tuple[float, float, float, float]]] = {}
    for page_index, page in enumerate(reader.pages):
        pid = page.indirect_reference.idnum
        wanted = {mcid: owner for (p, mcid), owner in owned.items() if p == pid}
        if not wanted:
            continue
        stack: list[int | None] = []
        boxes: dict[int, list[float]] = defaultdict(lambda: [1e9, 1e9, -1e9, -1e9])

        def before(op, args, cm, tm):
            if op == b"BDC":
                props = args[1] if len(args) > 1 else None
                mcid = props.get("/MCID") if hasattr(props, "get") else None
                stack.append(int(mcid) if mcid is not None else None)
            elif op == b"BMC":
                stack.append(None)
            elif op == b"EMC" and stack:
                stack.pop()

        def on_text(text, cm, tm, font, size):
            if not text.strip():
                return
            mcid = next((m for m in reversed(stack) if m is not None), None)
            owner = wanted.get(mcid)
            if owner is None:
                return
            x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            fs = (size or 10) * abs(cm[3] or 1) * abs(tm[3] or 1)
            b = boxes[owner]
            b[0], b[1] = min(b[0], x), min(b[1], y - 0.3 * fs)
            b[2], b[3] = max(b[2], x + 0.5 * fs * len(text)), max(b[3], y + fs)

        page.extract_text(visitor_operand_before=before, visitor_text=on_text)
        if boxes:
            result[page_index] = [(b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2) for b in boxes.values()]
    return result


# ---------------------------------------------------------------------------
# Numbering grammar
# ---------------------------------------------------------------------------


def successor_gap(prev: tuple[int, ...], cand: tuple[int, ...], max_gap: int = MAX_GAP,
                  class_coded: bool = True) -> int | None:
    """Numbers skipped when cand follows prev, or None if cand cannot follow prev.

    prev == () means nothing has been seen yet in the container. Gaps cover
    paragraphs deleted by amendments (12 -> 12.2) and levels without a heading
    of their own (3 -> 3.1.1)."""
    if not prev:
        if any(c != 1 for c in cand[:-1]):
            return None
        gap = len(cand) - 1 + cand[-1] - 1
        return gap if gap <= max_gap else None
    if len(cand) > len(prev) and cand[: len(prev)] == prev:
        extra = cand[len(prev):]
        if all(c == 1 for c in extra[:-1]) and extra[-1] >= 0:
            # children may start at 0 (ADN "9.3.3.0 Materials of construction")
            gap = len(extra) - 1 + max(extra[-1] - 1, 0)
            if gap <= max_gap:
                return gap
    for depth in range(min(len(prev), len(cand)), 0, -1):
        if cand[: depth - 1] != prev[: depth - 1]:
            continue
        gap = cand[depth - 1] - prev[depth - 1] - 1
        if (gap < 0 or gap > max_gap) and class_coded:
            gap = _class_coded_gap(prev[depth - 1], cand[depth - 1], gap)
        tail = cand[depth:]
        if tail:
            # next sibling of an ancestor, then down into it without headings for
            # the levels in between (ADN: 1.1.5 -> 1.2.1 when "CHAPTER 1.2" is missed)
            if gap != 0 or any(c != 1 for c in tail[:-1]):
                continue
            gap += len(tail) + tail[-1] - 1
        if 0 <= gap <= max_gap:
            return gap
    return None


def _class_coded_gap(prev: int, cand: int, gap: int) -> int:
    """Siblings numbered by class and division (ADN/ADR 2.2.3, 2.2.41, 2.2.42,
    2.2.43, 2.2.51, ..., 2.2.62, 2.2.7): read a two-digit number as its leading
    digit plus a division, so 3 -> 41, 43 -> 51 and 62 -> 7 are next siblings."""
    p, c = str(prev), str(cand)
    if len(p) <= 2 and len(c) <= 2 and len(p) + len(c) >= 3 and "0" not in p + c:
        # "1x" is read as an ordinal (10 -> 11 -> 2 would be a list running out)
        # divisions run 1-3 (4.1-4.3); "47 -> 5" is a numbered list, not a class change
        if int(c[0]) == int(p[0]) + 1 and c[1:] in ("", "1") and p[0] != "1" and p[1:] <= "3":
            return 0
    return gap


def _best_chain(items: list[dict], gap_penalty: float, free_start: bool = False,
                resync: bool = False) -> tuple[float, list[dict]]:
    """Maximum-weight chain of legal successors over items in document order.

    Each item carries val (tuple of ints), w (weight) and worded (text follows
    the number); only worded lines may skip numbers."""
    n = len(items)
    best, back = [0.0] * n, [-1] * n

    def edge(prev: tuple[int, ...], item: dict) -> float | None:
        gap = successor_gap(prev, item["val"])
        cand = item["val"]
        if (gap is None and prev and resync and item["worded"] and cand > prev
                and len(cand) > 1 and cand[0] - prev[0] <= 1):
            # resynchronise after a break in a long document (a heading lost in
            # extraction); numbering must still move forward, within the same or the
            # next top-level part, so a numbered list (ADN special provisions
            # 188, 189, ...) cannot be reached this way
            return item["w"] - RESYNC_COST
        if gap is None or (gap and not item["worded"]):
            return None
        if not item["worded"] and successor_gap(prev, cand, class_coded=False) is None:
            return None  # a bare number ("41" page number) may not continue a chain by class coding
        return item["w"] - gap_penalty * gap

    for j in range(n):
        start = edge((), items[j])
        if start is None and free_start and items[j]["worded"]:
            start = items[j]["w"] - gap_penalty  # a volume may open at any number
        if start is not None and start > 0:
            best[j] = start
        for i in range(j):
            if best[i]:
                score = edge(items[i].get("last", items[i]["val"]), items[j])
                if score is not None and best[i] + score > best[j]:
                    best[j], back[j] = best[i] + score, i
    if not n or not max(best):
        return 0.0, []
    j = max(range(n), key=best.__getitem__)
    total, chain = best[j], []
    while j != -1:
        chain.append(items[j])
        j = back[j]
    return total, chain[::-1]


def _roman_value(s: str) -> int | None:
    if len(s) == 1 and s in ROMAN_UNICODE:
        return ROMAN_UNICODE.index(s) + 1
    if not s or not re.fullmatch(r"X{0,3}(IX|IV|V?I{0,3})", s):
        return None
    vals = {"I": 1, "V": 5, "X": 10}
    total = 0
    for a, b in zip(s, s[1:] + " "):
        total += -vals[a] if b != " " and vals[a] < vals[b] else vals[a]
    return total


def _roman_label(v: int) -> str:
    out = ""
    for n, r in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while v >= n:
            out, v = out + r, v - n
    return out


def _scheme_candidates(text: str) -> list[tuple[str, tuple[int, ...], str, int]]:
    """Roman (R) / capital-letter (L) readings of a heading line: (kind, value, label, end).
    I, V and X are ambiguous and come back under both readings."""
    out = []
    r = ROMAN_RE.match(text)
    if r and _roman_value(r.group(1)):
        v = _roman_value(r.group(1))
        out.append(("R", (v,), _roman_label(v), r.end()))
    l = LETTER_RE.match(text)
    if l and WORD_RE.search(text[l.end():]):
        out.append(("L", (ord(l.group(1)) - 64,), l.group(1), l.end()))
    return out


def _container_label(match: re.Match, current: str | None) -> str:
    label = re.sub(r"\s*[-–]\s*", " - ", re.sub(r"\s+", " ", match.group(1)))
    if label.lower().startswith("appendix") and current:
        return f"{current.split(' - ')[0]} - {label}"
    return label


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def decode_structure(
    lines: list[Line],
    start: str = DEFAULT_START_RE,
    *,
    gap_penalty: float = 0.4,
    footnote_weight: float = 0.2,
    wordless_weight: float = 0.3,
    profile: dict | None = None,
) -> list[Heading]:
    """Decode the heading hierarchy of a document from its layout lines.

    start: regex for the line where the body begins (the table of contents is
    skipped automatically); by default the first bold line numbered "1"/"I.".
    profile: if given, receives the document-level decisions taken (arabic
    restart scope per container, sub-rule convention, note label)."""
    if not lines:
        return []
    sizes = [ln.size for ln in lines]
    body = max(set(sizes), key=sizes.count)
    top_size = max((ln.size for ln in lines if ln.bold), default=body)
    profile = profile if profile is not None else {}

    furniture = _page_furniture(lines)
    note_labels = _note_labels(lines, furniture, body)
    n_inline = sum(1 for ln in lines if re.match(r"^[A-Z]?\d+(\.\d+)+\s+\(1\)\s", ln.text))
    subrule_mode = "inline" if n_inline >= 3 else "any"
    profile.update(note_labels=sorted(note_labels), subrules=subrule_mode, scopes=[])

    segments: list[tuple[str | None, int | None, list[dict]]] = []
    current: tuple[str | None, int | None, list[dict]] = (None, None, [])
    started = not any(not TOC_RE.search(ln.text) and ln.bold and re.match(start, ln.text) for ln in lines)
    if started:
        logger.warning("No line matches the start pattern %r; decoding from the first line", start)
    pending = None
    for idx, ln in enumerate(lines):
        text = ln.text
        if text in furniture:
            continue
        if not started:
            started = not TOC_RE.search(text) and ln.bold and bool(re.match(start, text))
            if not started:
                continue
        v = VOLUME_RE.match(text)
        if v and ln.bold and ln.size >= body * 1.5:  # title-page size, not body text
            segments.append(current)
            current = (f"Volume {v.group(1)}", idx, [])
            pending = None
            continue
        a = ANNEX_RE.match(text)
        if a and ln.bold and ln.size >= top_size * 0.97:
            segments.append(current)
            current = (_container_label(a, current[0]), idx, [])
            pending = None
            continue
        ap = APP_RE.match(text)
        if ap and ln.bold and not TOC_RE.search(text):
            segments.append(current)
            current = (f"APP {ap.group(2)}", idx, [])
            pending = None
            continue
        if TOC_PAGE_RE.search(text):
            continue  # dot leaders followed by a page number: a table-of-contents entry
        items = current[2]
        w = 1.0 if ln.size >= body * 0.97 else footnote_weight
        pm = PAREN_RE.match(text)
        if pm and WORD_RE.search(text[pm.end():]):
            items.append(dict(kind="P", val=(int(pm.group(1)),), label=f"({pm.group(1)})", w=w,
                              worded=True, idx=idx, text=text[pm.end():].strip()))
            continue
        if ln.bold and text.strip() in note_labels:
            items.append(dict(kind="G", val=(), label=text.strip(), w=0, worded=True, idx=idx, text=""))
        for kind, val, lab, end in _scheme_candidates(text):
            items.append(dict(kind=kind, val=val, label=lab, w=w, worded=True, idx=idx,
                              text=text[end:].strip()))
        m = NUM_RE.match(text) or ALNUM_RE.match(text) or KEYWORD_NUM_RE.match(text)
        if m:
            cand = tuple(int(x) for x in re.sub(r"^[A-Z]", "", m.group(1)).split("."))
            rest = text[m.end():].strip()
            # a keyword heading ("CHAPTER 7.2") often has its title on the next line
            worded = bool(WORD_RE.search(rest)) or m.re is KEYWORD_NUM_RE
            if pending and len(cand) > 1 and cand[-1] == 1:
                # implicit parent: "Scenario 2" heading directly before "2.1"
                parent = cand[:-1]
                items.append(dict(kind="A", val=parent, label=".".join(map(str, parent)), w=1.0,
                                  worded=True, idx=pending[2], text=pending[1], virtual=True))
            pending = None
            item = dict(kind="A", val=cand, label=m.group(1), worded=worded, idx=idx, text=rest,
                        w=w if worded else w * wordless_weight, bold=ln.bold,
                        short=len(rest.split()) <= 16)
            r = RANGE_RE.match(rest)
            if r:
                end = tuple(int(x) for x in r.group(1).split("."))
                if len(end) == len(cand) and end[:-1] == cand[:-1] and end[-1] > cand[-1]:
                    item["last"] = end
            items.append(item)
        elif ln.bold and body * 1.05 < ln.size < top_size * 0.97 and len(text.split()) <= 10:
            if pending is None or ln.size > pending[0]:
                pending = (ln.size, text, idx)
        elif not (ln.bold and ln.size >= top_size * 0.97):
            pending = None
    segments.append(current)

    headings: list[Heading] = []
    for container, container_idx, items in segments:
        headings += _decode_segment(container, container_idx, items, lines, gap_penalty,
                                    subrule_mode, profile)
    return _unprefix_volumes(headings)


def _unprefix_volumes(headings: list[Heading]) -> list[Heading]:
    """Drop the "Volume II / " namespace when the volumes' ids do not collide:
    volumes split a document physically, but its numbering stays global, and
    cross-references ("see 2.2.8.1") cite the bare id."""
    vol = lambda h: h.clause_id.split(" / ")[0] if VOLUME_RE.match(h.clause_id.split(" / ")[0]) else None
    if not any(vol(h) for h in headings):
        return headings
    bare: dict[str, set] = defaultdict(set)
    for h in headings:
        if h.kind != "container":
            v = vol(h)
            bare[h.clause_id[len(v) + 3:] if v else h.clause_id].add(v)
    if any(len(vs) > 1 for vs in bare.values()):
        return headings
    out = []
    for h in headings:
        v = vol(h)
        if h.kind == "container" and v:
            continue
        out.append(Heading(h.clause_id[len(v) + 3:], h.label, h.line_index, h.text, h.kind) if v else h)
    return out


def _decode_segment(container, container_idx, items, lines, gap_penalty, subrule_mode, profile):
    prefix = f"{container} / " if container else ""
    free = bool(container and VOLUME_RE.match(container))
    out: list[Heading] = []
    if container:
        out.append(Heading(container, container, container_idx, "", "container"))

    _, romans = _best_chain([it for it in items if it["kind"] == "R"], gap_penalty)
    roman_idx = {it["idx"] for it in romans}
    bounds = [it["idx"] for it in romans]

    def roman_at(idx):
        label = None
        for it in romans:
            if it["idx"] <= idx:
                label = it["label"]
        return label

    letters = []
    for k in range(len(bounds) + 1):
        lo = bounds[k - 1] if k else -1
        hi = bounds[k] if k < len(bounds) else float("inf")
        seg = [it for it in items if it["kind"] == "L" and lo < it["idx"] < hi and it["idx"] not in roman_idx]
        letters += _best_chain(seg, gap_penalty)[1]
    taken = roman_idx | {it["idx"] for it in letters}

    def letter_at(idx):
        r, label = roman_at(idx), None
        for it in letters:
            if it["idx"] <= idx and roman_at(it["idx"]) == r:
                label = it["label"]
        return label

    for it in romans:
        out.append(Heading(prefix + it["label"], it["label"], it["idx"], it["text"], "roman"))
    for it in letters:
        r = roman_at(it["idx"])
        cid = f"{prefix}{r} / {it['label']}" if r else prefix + it["label"]
        out.append(Heading(cid, it["label"], it["idx"], it["text"], "letter"))

    # note blocks: claim the restarting "1., 2., ..." paragraphs after a note heading
    note_blocks = []
    ordered = sorted((it for it in items if it["kind"] in "AG" and not it.get("virtual")),
                     key=lambda it: it["idx"])
    for k, g in enumerate(ordered):
        if g["kind"] != "G":
            continue
        claimed, expected = [], 1
        for it in ordered[k + 1:]:
            if it["kind"] == "G" or len(it["val"]) != 1 or it.get("bold") or it["val"][0] != expected:
                break
            claimed.append(it)
            expected += 1
        note_blocks.append((g, claimed))
        taken |= {it["idx"] for it in claimed}

    # arabic clauses: keep the restart scope that explains the most numbering
    arabic = [it for it in items if it["kind"] == "A" and (it.get("virtual") or it["idx"] not in taken)]
    scopes = {
        "container": lambda idx: (),
        "roman": lambda idx: tuple(x for x in (roman_at(idx),) if x),
        "letter": lambda idx: tuple(x for x in (roman_at(idx), letter_at(idx)) if x),
    }
    best = None
    for name, key in scopes.items():
        groups = defaultdict(list)
        for it in arabic:
            groups[key(it["idx"])].append(it)
        # Every extra group is a fresh chain start, so splitting can only add
        # score; a restart scope has to pay for each restart it claims. Without
        # this a stray "I." / "II." pair (an ADN flowchart) splits a whole document.
        total, chosen = -RESTART_COST * max(0, len(groups) - 1), []
        for g, its in groups.items():
            score, chain = _best_chain(sorted(its, key=lambda it: it["idx"]), gap_penalty, free,
                                       resync=True)
            total += score
            chosen += [(g, it) for it in chain]
            if g and chain and chain[0]["val"][0] != 1:
                total = float("-inf")  # a part whose numbering does not restart at 1 is no restart
        if best is None or total > best[0] + 1e-9:
            best = (total, name, chosen)
    if arabic:
        profile["scopes"].append((container, best[1]))

    emitted: list[tuple[int, str]] = []
    for g, it in best[2]:
        cid = prefix + "".join(f"{x} / " for x in g) + it["label"]
        kind = "virtual" if it.get("virtual") else "clause"
        out.append(Heading(cid, it["label"], it["idx"], it["text"], kind))
        if kind == "clause":
            emitted.append((it["idx"], cid))

    # interleaved bold sub-heading sequence the main chain left over (GTR 21)
    used = {id(it) for _, it in best[2]}
    groups = defaultdict(list)
    for it in arabic:
        if id(it) not in used and it.get("bold") and it.get("short") and it["worded"]:
            groups[tuple(x for x in (roman_at(it["idx"]), letter_at(it["idx"])) if x)].append(it)
    for g, its in groups.items():
        _, chain = _best_chain(its, gap_penalty)
        if len(chain) >= 2:
            for it in chain:
                out.append(Heading(prefix + "".join(f"{x} / " for x in g) + it["label"], it["label"],
                                   it["idx"], it["text"], "subheading"))

    # "(n)" sub-rules between consecutive clauses
    emitted.sort()
    inline = {e for e in emitted if re.match(r"^\S+\s+\(1\)\s", lines[e[0]].text)}
    groups = defaultdict(list)
    for e in inline:
        rest = re.sub(r"^\S+\s+\(1\)\s+", "", lines[e[0]].text)
        groups[e].append(dict(kind="P", val=(1,), label="(1)", w=1.0, worded=True, idx=e[0], text=rest,
                              same_line=True))
    for it in items:
        if it["kind"] == "P" and it["idx"] not in taken:
            owner = max((e for e in emitted if e[0] < it["idx"]), default=None)
            if owner and (subrule_mode == "any" or owner in inline):
                groups[owner].append(it)
    for (oidx, oid), its in groups.items():
        for it in _best_chain(sorted(its, key=lambda it: it["idx"]), gap_penalty)[1]:
            out.append(Heading(f"{oid} / {it['label']}", it["label"], it["idx"], it["text"], "subrule"))

    # note blocks attach to the clause that precedes them
    for g, claimed in note_blocks:
        owner = max((e for e in emitted if e[0] < g["idx"]), default=None)
        base = f"{owner[1]} / {g['label']}" if owner else prefix + g["label"]
        out.append(Heading(base, g["label"], g["idx"], "", "note"))
        for it in claimed:
            out.append(Heading(f"{base} / {it['label']}", it["label"], it["idx"], it["text"], "clause"))
    return out


def _page_furniture(lines: list[Line]) -> set[str]:
    """Texts that are the first or last line on many pages (running header/footer)."""
    by_page: dict[int, list[str]] = defaultdict(list)
    for ln in lines:
        by_page[ln.page].append(ln.text)
    edge_pages: dict[str, set[int]] = defaultdict(set)
    for page, texts in by_page.items():
        for t in {texts[0], texts[-1]}:
            edge_pages[t].add(page)
    return {t for t, pages in edge_pages.items() if len(pages) >= max(3, 0.2 * len(by_page))}


def _note_labels(lines: list[Line], furniture: set[str], body: float) -> set[str]:
    """The dominant recurring short bold unnumbered line(s), e.g. ADGM's "Guidance".
    Smaller-than-body labels are figure annotations ("TTC" in R152's diagrams)."""
    counts: dict[str, int] = defaultdict(int)
    for ln in lines:
        t = ln.text.strip()
        if (ln.bold and ln.size >= body * 0.97 and t not in furniture and not NUM_RE.match(t) and not ALNUM_RE.match(t)
                and len(t.split()) <= 3 and t[:1].isupper() and not _scheme_candidates(t)
                and not ANNEX_RE.match(t)):
            counts[t] += 1
    top = max(counts.values(), default=0)
    return {t for t, c in counts.items() if c >= 3 and c >= 0.25 * top}
