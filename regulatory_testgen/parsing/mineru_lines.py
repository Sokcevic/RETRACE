"""Decoder input from MinerU: reading order and line breaks from MinerU, fonts from the PDF.

PyMuPDF (structure_decoder.pdf_layout_lines) groups spans by baseline across the
whole page, which merges the two columns of a two-column paper into one line.
MinerU's layout model returns blocks in reading order, column by column, and drops
running headers, footers and page numbers ("discarded_blocks"). This module turns
MinerU's ``<name>_middle.json`` into the same ``Line`` list the decoder expects:

* one Line per MinerU text line, in MinerU's reading order;
* tables, images and display equations are skipped (as in pdf_layout_lines);
* font size and weight are not in MinerU's output, so they are read from the PDF
  spans that fall inside the line's box; a line of a MinerU ``title`` block is
  also marked bold, so the layout cue survives when the PDF has no bold font.

Run MinerU first, e.g. ``mineru -p paper.pdf -o out/ -b pipeline -m auto``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .structure_decoder import INVISIBLE_RE, Line

SKIP = {"table", "image", "interline_equation", "equation"}


def _pdf_spans(pdf_path: Path) -> dict[int, list[tuple[float, float, float, bool]]]:
    """Per page: (center x, center y, size, bold) of every non-empty span."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    out: dict[int, list] = {}
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            spans = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for s in line["spans"]:
                        if s["text"].strip():
                            x0, y0, x1, y1 = s["bbox"]
                            bold = bool(s["flags"] & 16) or "bold" in s["font"].lower()
                            spans.append(((x0 + x1) / 2, (y0 + y1) / 2, s["size"], bold))
            out[i] = spans
    return out


def _blocks(block: dict):
    """A block and its nested blocks (lists, titles inside tables etc.), skipping non-text."""
    if block.get("type") in SKIP:
        return
    if block.get("lines"):
        yield block
    for sub in block.get("blocks", []):
        yield from _blocks(sub)


def mineru_layout_lines(pdf_path: str | Path, middle_json: str | Path) -> list[Line]:
    pdf_path, middle_json = Path(pdf_path), Path(middle_json)
    info = json.loads(middle_json.read_text())["pdf_info"]
    fonts = _pdf_spans(pdf_path)
    lines: list[Line] = []
    for page in info:
        p = page["page_idx"]
        spans = fonts.get(p, [])
        for top_block in page["para_blocks"]:
            for b in _blocks(top_block):
                title = b.get("type") == "title"
                for ln in b["lines"]:
                    text = " ".join(s.get("content", "") for s in ln["spans"]
                                    if s.get("type") in ("text", "inline_equation"))
                    text = INVISIBLE_RE.sub("", " ".join(text.split())).strip()
                    if not text:
                        continue
                    x0, y0, x1, y1 = ln["bbox"]
                    inside = [s for s in spans if x0 - 1 <= s[0] <= x1 + 1 and y0 - 1 <= s[1] <= y1 + 1]
                    size = round(max((s[2] for s in inside), default=10.0), 1)
                    bold = title or any(s[3] for s in inside)
                    lines.append(Line(page=p + 1, text=text, size=size, bold=bold, top=float(y0), x0=float(x0)))
    return lines


def find_middle_json(out_dir: str | Path, stem: str) -> Path | None:
    """MinerU writes <out>/<stem>/<method>/<stem>_middle.json."""
    hits = sorted(Path(out_dir).glob(f"{stem}/*/{stem}_middle.json"))
    return hits[0] if hits else None
