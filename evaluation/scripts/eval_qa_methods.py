"""Answer quality under matched context budgets: vanilla RAG, DOS RAG and RETRACE.

Contexts are built exactly as in eval_methods.py (same units, same budget rule) and
given to one open reader (Qwen3-32B via vLLM, see eval_qa.LLM).

Systems
  closed-book        no context
  vanilla            flat 200-word windows, dense retrieval, rank order
  vanilla-hybrid     flat windows, BM25 + dense (RRF), rank order
  dos-rag            vanilla units re-ordered by position in the document (Laitenberger et al. 2025)
  dos-rag-hybrid     vanilla-hybrid units in document order
  retrace            paragraph units (decoded clauses with heading path), hybrid, rank order
  retrace-docorder   retrace units in document order
  oracle             the gold text (ADN: cited clauses incl. sub-clauses, QASPER: gold evidence)

Metrics: ADN-QA multiple-choice accuracy (all / cited questions / basic questions),
QASPER official answer token F1. Paired bootstrap of every system against vanilla.

Usage: python eval_qa_methods.py <adn|qasper> [--budgets 1000 4000] [--limit N] [--model qwen3-32b]
  -> runs/qa_methods_<ds>_<model>.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import eval_methods as em  # noqa: E402
from eval_qa import LLM, MCQ_SYSTEM, QASPER_SYSTEM, fmt_mcq  # noqa: E402
from paths import ADNQA, W  # noqa: E402
from rag_core import encode_queries, flat_chunks, structure_chunks  # noqa: E402
from significance import bootstrap  # noqa: E402

SYSTEMS = [("vanilla", "flat-200", "dense", False), ("vanilla-hybrid", "flat-200", "hybrid", False),
           ("dos-rag", "flat-200", "dense", True), ("dos-rag-hybrid", "flat-200", "hybrid", True),
           ("retrace", "paragraph", "hybrid", False), ("retrace-docorder", "paragraph", "hybrid", True)]
THREADS = 32


def render(docs, prefix_of, c) -> str:
    body = docs[c.doc].text[c.start:c.end].strip()
    p = prefix_of.get((c.doc, c.node), "") if c.node else ""
    return (p + body) if p else body


def build(ds: str, budgets):
    em.DS = ds
    docs, rq = em.LOADERS[ds](None)
    if ds == "adn":
        from eval_paragraph import paragraph_units, struct_gold, unit_text
        doc = docs["adn"]
        units = paragraph_units(doc, structure_chunks(doc))
        qa = [json.loads(l) for l in open(ADNQA / "qa.jsonl")]
        test = set(json.load(open(ADNQA / "splits.json"))["test"])
        items = [q for q in qa if q["id"] in test]
        queries = [{"id": q["id"], "scope": None, "query": q["question"] + " " + " ".join(q["options"].values()),
                    "gold": struct_gold(doc, units, q["adn_refs"]) if q["adn_refs"] else []} for q in items]

        def oracle(q):
            out = []
            for g in q["adn_refs"]:
                ids = [p for p in units if p and (p == g or p.startswith(g + "."))]
                if ids:
                    out.append(unit_text(doc, {"prefix": units[ids[0]]["prefix"],
                                               "spans": [sp for p in ids for sp in units[p]["spans"]]}, 1000))
            return out
        oracles = [oracle(q) for q in items]
    else:
        items = queries = rq
        oracles = [[docs[g[0]].text[g[1]:g[2]] for g in q["gold"]] for q in queries]
    flat = em.System("flat-200", docs, [c for d in docs.values() for c in em.with_size(200, flat_chunks, d)])
    s200 = [c for d in docs.values() for c in structure_chunks(d)]
    para = em.System("paragraph", docs, s200, paragraph=True)
    prefix_of = {}
    for c in s200:
        body = docs[c.doc].text[c.start:c.end]
        if c.text.endswith(body) and len(c.text) > len(body):
            prefix_of.setdefault((c.doc, c.node), c.text[: len(c.text) - len(body)])
    systems = {"flat-200": flat, "paragraph": para}
    qvecs = encode_queries([q["query"] for q in queries])
    ctx = {f"closed-book": [[] for _ in queries], "oracle": oracles}
    for name, sysname, ret, docorder in SYSTEMS:
        s = systems[sysname]
        ranked = [s.ranked(q, qv, ret) for q, qv in zip(queries, qvecs)]
        for B in budgets:
            out = []
            for r in ranked:
                cs = s.context(r, B)
                if docorder:
                    cs = sorted(cs, key=lambda c: (c.doc, c.start))
                out.append([render(docs, prefix_of, c) for c in cs])
            ctx[f"{name}@{B}"] = out
    return items, queries, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=("adn", "qasper"))
    ap.add_argument("--budgets", type=int, nargs="+", default=[1000, 4000])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default="qwen3-32b")
    args = ap.parse_args()
    items, queries, ctx = build(args.dataset, args.budgets)
    if args.limit:
        items, queries = items[: args.limit], queries[: args.limit]
    llm = LLM(args.model)
    per, report = {}, {}

    if args.dataset == "adn":
        def score(system, i):
            q = items[i]
            if system == "oracle" and not ctx[system][i]:
                return None
            a = llm(MCQ_SYSTEM, fmt_mcq(q, ctx[system][i]), max_tokens=5)
            m = re.search(r"\b([ABCD])\b", a.strip().upper())
            return float((m.group(1) if m else "?") == q["answer"])
    else:
        sys.path.insert(0, str(W / "raw/qasper"))
        from qasper_evaluator import token_f1_score

        def golds(q):
            out = []
            for a in q["answers"]:
                a = a["answer"]
                if a["unanswerable"]:
                    out.append("Unanswerable")
                elif a["extractive_spans"]:
                    out.append(", ".join(a["extractive_spans"]))
                elif a["free_form_answer"]:
                    out.append(a["free_form_answer"])
                elif a["yes_no"] is not None:
                    out.append("Yes" if a["yes_no"] else "No")
            return out

        def score(system, i):
            q = queries[i]
            user = "Excerpts:\n\n" + "\n\n---\n\n".join(ctx[system][i]) + f"\n\nQuestion: {q['query']}\nAnswer:"
            pred = llm(QASPER_SYSTEM, user, max_tokens=120).strip()
            return max(token_f1_score(pred, g) for g in golds(q))

    order = ["closed-book"] + [f"{n}@{B}" for B in args.budgets for n, *_ in SYSTEMS] + ["oracle"]
    for system in order:
        with cf.ThreadPoolExecutor(THREADS) as ex:
            vals = list(ex.map(lambda i: score(system, i), range(len(queries))))
        per[system] = vals
        v = np.array([np.nan if x is None else x for x in vals], float)
        r = {"score": round(float(np.nanmean(v)), 4), "n": int((~np.isnan(v)).sum()),
             "context_words": round(float(np.mean([sum(len(t.split()) for t in c) for c in ctx[system]])), 1)}
        if args.dataset == "adn":
            cited = np.array([bool(q["adn_refs"]) for q in items])
            r["cited"] = round(float(np.nanmean(v[cited])), 4)
            r["basic"] = round(float(np.nanmean(v[~cited])), 4) if (~cited).any() else None
        report[system] = r
        print(system, r, flush=True)
    for B in args.budgets:
        base = np.array(per[f"vanilla@{B}"], float)
        for n, *_ in SYSTEMS[1:]:
            report[f"{n}@{B}"]["vs vanilla"] = bootstrap(np.array(per[f"{n}@{B}"], float), base, 10000)
        report[f"retrace@{B}"]["vs dos-rag-hybrid"] = bootstrap(np.array(per[f"retrace@{B}"], float),
                                                              np.array(per[f"dos-rag-hybrid@{B}"], float), 10000)
    import rag_core
    out = W / f"runs/qa_methods_{args.dataset}{rag_core.tag()}_{args.model}.json"
    out.write_text(json.dumps({"model": args.model, "metric": "accuracy" if args.dataset == "adn" else "answer_f1",
                               "budgets_words": args.budgets, "results": report,
                               "ids": [q["id"] for q in queries], "per_item": per}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
