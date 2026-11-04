"""How structured are the four evaluation corpora, and how well does the decoder
recover that structure? Also: how often is the gold section cross-referenced
from elsewhere in its document?

Per dataset:
  docs, words, decoded headings per doc, docs without any heading, depth,
  share of words that sit under a decoded heading (not front matter),
  numbering style of the decoded headings, internal cross-references
  QASPER only: heading recall/precision against the section names in the
  dataset (the authors' LaTeX sections), matched by fuzzy title
Per query (runs/xref_<ds>.json, joined with retrieval hits later):
  the gold section(s) and how many sections of the same document refer to them

Usage: python analyze_structure.py [adn obliqa lbrag qasper]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rag_core  # noqa: E402
from rag_core import structure_chunks  # noqa: E402
from eval_retrieval import LOADERS, W  # noqa: E402

# a cross-reference: "Section 5.2", "Art. 4", "Rule 3.2.1", "§ 7", or a bare dotted number "7.2.4.1"
KEYWORD_REF = re.compile(r"\b(?:Sections?|Sec\.|Articles?|Art\.|Rules?|Chapters?|Clauses?|Paragraphs?|para\.|§)\s*"
                         r"(\d+(?:\.\d+)*)", re.I)
DOTTED_REF = re.compile(r"(?<![\d.])(\d+(?:\.\d+){2,6})(?![\d.]*\d)")  # 3+ levels; "1.5" is usually a quantity


def refs_in(text: str) -> set[str]:
    return set(KEYWORD_REF.findall(text)) | set(DOTTED_REF.findall(text))


def style(label: str) -> str:
    if re.fullmatch(r"\d+(\.\d+)+\.?", label):
        return "dotted"
    if re.fullmatch(r"\d+\.?", label):
        return "single number"
    if re.fullmatch(r"\(?[ivxlcdm]+[.)]?", label, re.I):
        return "roman"
    if re.fullmatch(r"\(?[a-z][.)]?", label, re.I):
        return "letter"
    return "other"


def norm_title(s: str) -> str:
    s = re.sub(r"^[\dIVXivx.()\s]+", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def analyse(ds: str) -> dict:
    from rapidfuzz import fuzz
    from regulatory_testgen.parsing.structure_decoder import decode_structure
    docs, queries = LOADERS[ds](None)
    if ds == "adn":  # gold from the decoded structure (see eval_paragraph.struct_gold)
        from eval_paragraph import paragraph_units, struct_gold, DS
        doc = docs["adn"]
        units = paragraph_units(doc, structure_chunks(doc))
        rows = {r["id"]: r for r in map(json.loads, open(DS / "retrieval.jsonl"))}
        for q in queries:
            q["gold"] = struct_gold(doc, units, rows[q["id"]]["gold"])
    per_doc, styles, depths = [], Counter(), Counter()
    sec_of: dict[str, list] = {}  # doc -> [(start, end, node)]
    indeg: dict[str, Counter] = {}
    qasper_match = []
    qasper_gold = json.load(open(W / "raw/qasper/qasper-test-v0.3.json")) if ds == "qasper" else {}
    for did, doc in docs.items():
        heads = decode_structure(doc.lines)
        chunks = structure_chunks(doc)
        words = len(doc.text.split())
        ids = {c.node for c in chunks if c.node and c.node != "front-matter"}
        for h in heads:
            styles[style(h.label)] += 1
            depths[h.clause_id.split(" / ")[-1].count(".") + 1] += 1
        # contiguous runs of each section (a clause number can occur twice, e.g. in two volumes)
        segs: list[list] = []
        for c in sorted(chunks, key=lambda c: c.start):
            if segs and segs[-1][2] == c.node and c.start <= segs[-1][1] + 1:
                segs[-1][1] = max(segs[-1][1], c.end)
            else:
                segs.append([c.start, c.end, c.node])
        sec_of[did] = [tuple(x) for x in segs]
        under = words - sum(len(doc.text[a:b].split()) for a, b, n in segs if n == "front-matter")
        # in-degree: number of distinct sections that cite a section
        citing = defaultdict(set)
        for a, b, n in segs:
            for r in refs_in(doc.text[a:b]):
                if r in ids and r != n:
                    citing[r].add(n)
        cnt = Counter({r: len(v) for r, v in citing.items()})
        indeg[did] = cnt
        per_doc.append({"doc": did, "words": words, "headings": len(heads), "under_heading": under / max(1, words),
                        "xrefs": sum(cnt.values()), "referenced_sections": len(cnt)})
        if ds == "qasper":
            gold = [norm_title(s["section_name"].split(":::")[-1]) for s in qasper_gold[did]["full_text"]
                    if s["section_name"]]
            gold = [g for g in dict.fromkeys(gold) if g]
            dec = [norm_title(h.text) for h in heads]
            # two-column PDFs merge the heading line with text of the other column,
            # so compare the gold title with the start of the decoded line
            same = lambda g, d: fuzz.ratio(g, d[:len(g)]) >= 85
            hit = [any(same(g, d) for d in dec) for g in gold]
            prec = [any(same(g, d) for g in gold) for d in dec]
            qasper_match.append((sum(hit), len(gold), sum(prec), len(dec)))
    # per query: gold sections and their in-degree
    qrows = []
    for q in queries:
        nodes = set()
        for did, s, e in q["gold"]:
            for a, b, n in sec_of[did]:
                if min(b, e) - max(a, s) > 0.5 * min(b - a, e - s):
                    nodes.add((did, n))
        deg = [sum(v for k, v in indeg[d].items() if n not in (None, "front-matter") and (k == n or k.startswith(n + ".")))
               for d, n in nodes]
        qrows.append({"id": q["id"], "gold_sections": sorted(n for _, n in nodes if n),
                      "in_front_matter": any(n == "front-matter" for _, n in nodes),
                      "indegree": max(deg) if deg else 0})
    (W / f"runs/xref_{ds}{rag_core.tag()}.json").write_text(json.dumps(qrows))
    pd = per_doc
    out = {
        "docs": len(pd),
        "words_median": int(np.median([d["words"] for d in pd])),
        "headings_per_doc_median": float(np.median([d["headings"] for d in pd])),
        "docs_without_headings": sum(d["headings"] == 0 for d in pd),
        "share_words_under_heading": round(float(np.mean([d["under_heading"] for d in pd])), 3),
        "depth": dict(sorted(depths.items())),
        "styles": dict(styles.most_common()),
        "xrefs_per_doc_median": float(np.median([d["xrefs"] for d in pd])),
        "share_sections_referenced": round(sum(d["referenced_sections"] for d in pd) /
                                           max(1, sum(d["headings"] for d in pd)), 3),
        "queries": len(qrows),
        "queries_gold_referenced": round(float(np.mean([r["indegree"] > 0 for r in qrows])), 3),
        "queries_gold_in_front_matter": round(float(np.mean([r["in_front_matter"] for r in qrows])), 3),
    }
    if qasper_match:
        h, g, p, d = map(sum, zip(*qasper_match))
        out["qasper_heading_recall"] = round(h / g, 3)
        out["qasper_heading_precision"] = round(p / max(1, d), 3)
    return out


def main():
    dss = sys.argv[1:] or ["adn", "obliqa", "lbrag", "qasper"]
    path = W / "runs/structure_analysis.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    for ds in dss:
        report[ds + rag_core.tag()] = analyse(ds)
        print(ds + rag_core.tag(), json.dumps(report[ds + rag_core.tag()]))
        path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
