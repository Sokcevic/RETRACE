"""Paired bootstrap significance tests for the RETRACE evaluation.

Retrieval: per-query hit@1, hit@5, MRR@10 recomputed from runs/hits_*.pkl, every
system compared with the flat baseline (same retriever, same queries).
QA: per-question correctness / F1 from runs/qa_items_*.json, every system
compared with flat.

For each pair: mean difference, 95% percentile CI over B resamples of queries,
and a two-sided p-value (2 * share of resamples on the other side of zero).
Usage: python significance.py [--b 10000]  -> runs/significance.json + table on stdout
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rag_core import Chunk, is_hit  # noqa: E402

from paths import W  # noqa: E402
SEED = 13


def bootstrap(a: np.ndarray, b: np.ndarray, B: int) -> dict:
    """Paired bootstrap of mean(a - b)."""
    d = a - b
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"delta": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p": round(float(min(1.0, p)), 4), "n": int(len(d))}


def per_query_retrieval(pkl: Path) -> dict:
    data = pickle.load(open(pkl, "rb"))
    queries = data["queries"]
    out = {}
    for system, chunks in data["systems"].items():
        chunks = [Chunk(doc, s, e, t, node) for doc, s, e, t, node in chunks]
        for ret, tops in data["hits"][system].items():
            h1, h5, rr = [], [], []
            for q, top in zip(queries, tops):
                first = next((r for r, i in enumerate(top) if any(is_hit(chunks[i], g) for g in q["gold"])), None)
                h1.append(first is not None and first < 1)
                h5.append(first is not None and first < 5)
                rr.append(1 / (first + 1) if first is not None else 0.0)
            out[(system, ret)] = {"hit@1": np.array(h1, float), "hit@5": np.array(h5, float),
                                  "mrr@10": np.array(rr, float)}
    # structure+graph re-uses the structure chunk list
    for ret, tops in data["hits"].get("structure+graph", {}).items():
        chunks = [Chunk(doc, s, e, t, node) for doc, s, e, t, node in data["systems"]["structure"]]
        h1, h5, rr = [], [], []
        for q, top in zip(queries, tops):
            first = next((r for r, i in enumerate(top) if any(is_hit(chunks[i], g) for g in q["gold"])), None)
            h1.append(first is not None and first < 1)
            h5.append(first is not None and first < 5)
            rr.append(1 / (first + 1) if first is not None else 0.0)
        out[("structure+graph", ret)] = {"hit@1": np.array(h1, float), "hit@5": np.array(h5, float),
                                         "mrr@10": np.array(rr, float)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=10000)
    args = ap.parse_args()
    report = {}
    for pkl in sorted(glob.glob(str(W / "runs/hits_*.pkl"))):
        name = Path(pkl).stem.replace("hits_", "")
        per = per_query_retrieval(Path(pkl))
        rows = {}
        for (system, ret), m in per.items():
            if system == "flat":
                continue
            base = per[("flat", ret)]
            rows[f"{system} vs flat [{ret}]"] = {k: bootstrap(m[k], base[k], args.b) for k in m}
        report[f"retrieval:{name}"] = rows
    for f in sorted(glob.glob(str(W / "runs/qa_items_*.json"))):
        items = json.load(open(f))
        name = Path(f).stem.replace("qa_items_", "")
        metric = items["metric"]
        base = items["items"]["flat"]
        rows = {}
        for system, vals in items["items"].items():
            if system == "flat":
                continue
            keep = [i for i, (a, b) in enumerate(zip(vals, base)) if a is not None and b is not None]
            rows[f"{system} vs flat"] = {metric: bootstrap(np.array([vals[i] for i in keep], float),
                                                           np.array([base[i] for i in keep], float), args.b)}
        report[f"qa:{name}"] = rows
    (W / "runs/significance.json").write_text(json.dumps(report, indent=1))
    for exp, rows in report.items():
        print(f"\n== {exp}")
        for pair, metrics in rows.items():
            if "[bm25]" in pair or "[dense]" in pair:
                continue
            cells = "  ".join(f"{k} {v['delta']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] p={v['p']:.3f}"
                              for k, v in metrics.items())
            print(f"  {pair:38s} {cells}")


if __name__ == "__main__":
    main()
