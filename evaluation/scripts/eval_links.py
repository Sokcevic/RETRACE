"""Ablation: how many cross-reference links to follow per retrieved clause.

RETRACE paragraph units (hybrid retrieval, local models). After each retrieved
clause, up to L clauses it cites are inserted into the ranking; the context is
then filled to the word budget as in eval_methods.py. L = 0 is plain RETRACE.
Reported separately for queries whose gold clause is cited by another clause
(the only ones links can help) and for the rest.

Usage: python eval_links.py <adn|obliqa> -> runs/links_<ds>.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import eval_methods as em  # noqa: E402
import rag_core  # noqa: E402
from rag_core import encode_queries, is_hit, structure_chunks  # noqa: E402
from significance import bootstrap  # noqa: E402

LINKS = (0, 1, 2, 3, 5)
BUDGETS = (1000, 2000, 4000)


def main():
    ds = sys.argv[1]
    em.DS = ds
    docs, queries = em.LOADERS[ds](None)
    if ds == "adn":
        from eval_paragraph import DS as ADNQA, paragraph_units, struct_gold
        doc = docs["adn"]
        units = paragraph_units(doc, structure_chunks(doc))
        rows = {r["id"]: r for r in map(json.loads, open(ADNQA / "retrieval.jsonl"))}
        for q in queries:
            q["gold"] = struct_gold(doc, units, rows[q["id"]]["gold"])
    chunks = [c for d in docs.values() for c in structure_chunks(d)]
    para = em.System("paragraph", docs, chunks, paragraph=True)
    first = {}
    refs = defaultdict(list)
    for i, c in enumerate(chunks):
        first.setdefault((c.doc, c.node), i)
        for r in c.refs:
            if r not in refs[(c.doc, c.node)]:
                refs[(c.doc, c.node)].append(r)
    xr = {r["id"]: r for r in json.load(open(em.W / f"runs/xref_{ds}.json"))}
    cited = np.array([xr[q["id"]]["indegree"] > 0 for q in queries])
    qvecs = encode_queries([q["query"] for q in queries])
    res, per = {}, {}
    for L in LINKS:
        hits = {B: [] for B in BUDGETS}
        for q, qv in zip(queries, qvecs):
            ranked, seen = [], set()
            for i in para.ranked(q, qv, "hybrid"):
                # every retrieved piece stays in the ranking; System.context applies the
                # paragraph rule (short clause whole, long clause piece by piece), so L = 0
                # equals RETRACE in eval_methods.py. Links are inserted once per clause.
                ranked.append(i)
                key = (chunks[i].doc, chunks[i].node)
                if key in seen:
                    continue
                seen.add(key)
                for r in refs[key][:L]:
                    j = first.get((key[0], r))
                    if j is not None:
                        ranked.append(j)
            for B in BUDGETS:
                ctx = para.context(ranked, B)
                hits[B].append(any(is_hit(c, g) for c in ctx for g in q["gold"]))
        res[f"L={L}"] = {}
        per[L] = {B: np.array(hits[B], float) for B in BUDGETS}
        for B in BUDGETS:
            h = per[L][B]
            res[f"L={L}"][f"hit@{B}w"] = {"all": round(h.mean(), 3), "gold cited": round(h[cited].mean(), 3),
                                           "gold not cited": round(h[~cited].mean(), 3)}
            if L:
                res[f"L={L}"][f"hit@{B}w"]["vs L=0"] = bootstrap(per[L][B], per[0][B], 10000)
        print(f"L={L}", json.dumps(res[f"L={L}"]), flush=True)
    res["n"] = {"all": len(queries), "gold cited": int(cited.sum())}
    (em.W / f"runs/links_{ds}{rag_core.tag()}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
