"""Build the ADN 2025 paragraph corpus from the official PDF with the RETRACE decoder.

Output: one JSON line per paragraph {id, title, text, references, parent}.
Volume namespaces are dropped (ADN numbering is global); where both volumes carry
the same id (Part headings, "3.2.4 (See Volume I)") the longer text is kept.
Usage: python build_adn_corpus.py <ADN 2025 English.pdf> <out.jsonl>
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from regulatory_testgen.parsing.pdf_parser import parse_pdf_clauses
from regulatory_testgen.structure_ids import parent_of

pdf, out = Path(sys.argv[1]), Path(sys.argv[2])
clauses = parse_pdf_clauses(pdf)
best: dict[str, dict] = {}
for c in clauses:
    d = c.model_dump()
    cid = re.sub(r"^Volume [IVX]+ / ", "", d["clause_id"])
    if cid.startswith("Volume ") or cid == "front-matter":
        continue
    rec = {"id": cid, "title": (d.get("title") or "").strip(), "text": (d.get("text") or "").strip(),
           "references": d.get("references") or []}
    if cid not in best or len(rec["text"]) > len(best[cid]["text"]):
        best[cid] = rec
for cid, rec in best.items():
    p = parent_of(cid)
    while p and p not in best:
        p = parent_of(p)
    rec["parent"] = p
with out.open("w") as f:
    for rec in best.values():
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(len(clauses), "clauses ->", len(best), "paragraphs")
