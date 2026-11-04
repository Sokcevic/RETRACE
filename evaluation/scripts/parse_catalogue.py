"""Parse an ADN catalogue-of-questions .docx into question records.

Each catalogue is a series of tables, one per examination objective:

    | <section title> / Examination objective N: <topic>          |
    | Number      | Source                    | Correct answer   |
    | 331 01.0-01 | Basic general knowledge   | B                |
    |             | <question> / A <opt> / B <opt> / C / D       |

The number, source and answer sit on one row, the question with its four
options on the next.  Questions are identical in structure across the
English, French, German and Russian catalogues, so the number is a stable
cross-language id.

Usage: python -I parse_catalogue.py <file.docx> [<file.docx> ...] > out.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn

QID = re.compile(r"^\s*(\d{3})\s+(\d{2})\.(\d)-(\d{2})\s*$")
# Option prefix A-D followed by a tab or spaces. The Russian catalogues type
# some letters as Cyrillic look-alikes (А, В, С), so those count as Latin.
OPTION = re.compile(r"^\s*([A-DАВС])[\t ]+(.*\S)\s*$")
CYR = {"А": "A", "В": "B", "С": "C"}
# ADN paragraph references: 7.2.4.16.4, 1.2.1, 3.2.3.2, 9.3.2.25.8 (b), 5.4.3
REF = re.compile(r"(?<![\d.])([1-9]\.\d{1,2}(?:\.\d{1,2}){0,5})(?:\s*\(([a-z]{1,2}|\d{1,2})\))?")
DELETED = re.compile(r"\b(deleted|supprim|gestrichen|исключ)", re.I)


def cell_paragraphs(cell) -> list[str]:
    return [p.text for p in cell.paragraphs]


def cell_has_image(cell) -> bool:
    xml = cell._tc.xml
    return "<w:drawing" in xml or "<v:imagedata" in xml or "<w:object" in xml


def parse_refs(source: str) -> list[str]:
    """ADN paragraph ids cited in a Source cell, e.g. '8.1.2.1 (g), 7.2.3.7.1.5'
    -> ['8.1.2.1', '7.2.3.7.1.5'].  Sub-items '(g)' are kept separately by the
    caller via parse_refs_detailed; the graph indexes paragraphs, not items."""
    return list(dict.fromkeys(m.group(1) for m in REF.finditer(source)))


def parse_refs_detailed(source: str) -> list[dict]:
    return [{"paragraph": m.group(1), "item": m.group(2)} for m in REF.finditer(source)]


def objective_of(header: str) -> tuple[str, str]:
    lines = [l.strip() for l in header.replace("\t", " ").split("\n") if l.strip()]
    section = lines[0] if lines else ""
    objective = lines[1] if len(lines) > 1 else ""
    return section, objective


def parse_question_cell(lines: list[str]) -> tuple[str, dict[str, str]]:
    stem: list[str] = []
    options: dict[str, str] = {}
    last = None
    # several options typed into one paragraph: "A\t... В\t... С\t..."
    lines = [part for raw in lines for part in re.split(r"[ \n]+(?=[A-DАВС]\t)", raw)]
    for raw in lines:
        if not raw.strip():
            continue
        m = OPTION.match(raw)
        if m and stem:
            letter = CYR.get(m.group(1), m.group(1))
            expected = "ABCD"[len(options)] if len(options) < 4 else None
            if letter == expected:
                options[letter] = m.group(2).strip()
                last = letter
                continue
        if last:  # continuation of an option
            options[last] += " " + raw.strip()
        else:
            stem.append(raw.strip())
    text = " ".join(stem).strip()
    if not options:
        # options run on inside one paragraph: "... done ?A\tInform ... B\tInform ..."
        parts = re.split(r"(?:(?<=\s)|(?<=[?:.]))([A-DАВС])\t", text)
        letters = [CYR.get(parts[k], parts[k]) for k in (1, 3, 5, 7)] if len(parts) == 9 else []
        if letters == list("ABCD"):
            return parts[0].strip(), {letters[n]: parts[k + 1].strip() for n, k in enumerate((1, 3, 5, 7))}
    return text, options


def parse_catalogue(path: Path) -> list[dict]:
    d = docx.Document(str(path))
    out: list[dict] = []
    for t in d.tables:
        rows = t.rows
        section = objective = ""
        if rows:
            first = rows[0].cells[0].text
            if not QID.match(first) and "\n" in first.strip():
                section, objective = objective_of(first)
        i = 0
        while i < len(rows):
            cells = rows[i].cells
            # Merged cells repeat; take distinct cells left to right.
            uniq = []
            for c in cells:
                if not uniq or c._tc is not uniq[-1]._tc:
                    uniq.append(c)
            m = QID.match(uniq[0].text) if uniq else None
            if not m or len(uniq) < 3:
                i += 1
                continue
            qid = uniq[0].text.strip()
            qid = re.sub(r"\s+", " ", qid)
            source = " ".join(uniq[1].text.split())
            answer = CYR.get(uniq[-1].text.strip(), uniq[-1].text.strip())
            # The question usually fills the next row; some catalogues put the
            # options in a further row, so read every row up to the next number.
            paras, has_image = [], False
            i += 1
            while i < len(rows) and not QID.match(rows[i].cells[0].text):
                cells = rows[i].cells
                qcell = max(cells, key=lambda c: len(c.text))
                paras += cell_paragraphs(qcell)
                has_image |= any(cell_has_image(c) for c in cells)
                i += 1
            stem, options = parse_question_cell(paras)
            deleted = bool(DELETED.search(source)) or (not stem and not options)
            out.append({
                "id": qid,
                "catalogue_section": section,
                "objective": objective,
                "source": source,
                "adn_refs": parse_refs(source),
                "adn_refs_detailed": parse_refs_detailed(source),
                "question": stem,
                "options": options,
                "answer": answer if answer in "ABCD" and answer else None,
                "has_image": has_image,
                "deleted": deleted,
            })
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        for rec in parse_catalogue(Path(p)):
            rec["file"] = Path(p).name
            print(json.dumps(rec, ensure_ascii=False))
