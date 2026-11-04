"""Re-score the chunk-level ADN runs (runs/hits_adn*.pkl) and the oracle QA
against the structure-derived gold spans of eval_paragraph.struct_gold, which
replace the line-based locator (it could land on paragraph numbers quoted in tables).
Usage: python rescore_adn.py -> runs/rescore_adn.json
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rag_core import Chunk, is_hit, structure_chunks  # noqa: E402
from eval_paragraph import DS, W, paragraph_units, struct_gold, unit_text  # noqa: E402
from eval_retrieval import load_adn  # noqa: E402
from significance import bootstrap  # noqa: E402


def main():
    docs, _ = load_adn(None)
    doc = docs["adn"]
    units = paragraph_units(doc, structure_chunks(doc))
    rows = {r["id"]: r for r in map(json.loads, open(DS / "retrieval.jsonl"))}
    report = {}
    for pkl, name in ((W / "runs/hits_adn.pkl", "bge"), (W / "runs/hits_adn_openai.pkl", "openai")):
        data = pickle.load(open(pkl, "rb"))
        golds = [struct_gold(doc, units, rows[q["id"]]["gold"]) for q in data["queries"]]
        per = {}
        for system, tops_by_ret in data["hits"].items():
            chunks = [Chunk(*c) for c in data["systems"]["structure" if system == "structure+graph" else system]]
            h1, h5, rr = [], [], []
            for g, top in zip(golds, tops_by_ret["hybrid"]):
                first = next((k for k, i in enumerate(top) if any(is_hit(chunks[i], x) for x in g)), None)
                h1.append(first is not None and first < 1)
                h5.append(first is not None and first < 5)
                rr.append(1 / (first + 1) if first is not None else 0.0)
            per[system] = {"hit@1": np.array(h1, float), "hit@5": np.array(h5, float), "mrr@10": np.array(rr, float)}
        report[name] = {s: {k: round(float(v.mean()), 4) for k, v in m.items()} for s, m in per.items()}
        report[name]["vs flat"] = {s: {k: bootstrap(per[s][k], per["flat"][k], 10000) for k in per[s]}
                                   for s in per if s != "flat"}
    # oracle QA with the corrected gold paragraphs (each capped at 1000 words)
    from eval_qa import LLM, MCQ_SYSTEM, fmt_mcq
    llm = LLM("gpt-4.1-mini")
    qa = [json.loads(l) for l in open(DS / "qa.jsonl")]
    test = set(json.load(open(DS / "splits.json"))["test"])
    qa = [q for q in qa if q["id"] in test and q["adn_refs"]]

    def ctx(q):
        out = []
        for g in q["adn_refs"]:
            ids = [p for p in units if p and (p == g or p.startswith(g + "."))]
            if ids:
                out.append(unit_text(doc, {"prefix": units[ids[0]]["prefix"],
                                           "spans": [sp for p in ids for sp in units[p]["spans"]]}, 1000))
        return out

    def one(q):
        c = ctx(q)
        if not c:
            return None
        a = llm(MCQ_SYSTEM, fmt_mcq(q, c), max_tokens=5)
        m = re.search(r"\b([ABCD])\b", a.strip().upper())
        return float((m.group(1) if m else "?") == q["answer"])
    with cf.ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(one, qa))
    vals = [o for o in ok if o is not None]
    report["oracle_qa"] = {"acc_cited": round(float(np.mean(vals)), 4), "n": len(vals), "n_cited": len(qa)}
    (W / "runs/rescore_adn.json").write_text(json.dumps(report, indent=1))
    for name in ("bge", "openai"):
        print(name, {s: v for s, v in report[name].items() if s != "vs flat"})
        for s, m in report[name]["vs flat"].items():
            print("  ", s, {k: (v["delta"], v["p"]) for k, v in m.items()})
    print(report["oracle_qa"])


if __name__ == "__main__":
    main()
