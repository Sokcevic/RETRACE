"""Retrieval evaluation: flat vs. structure (RETRACE) chunking on four datasets.

Usage: python eval_retrieval.py <dataset> [--limit N]
  dataset: adn | obliqa | lbrag | qasper
Writes runs/retrieval_<dataset>.json (metrics) and runs/hits_<dataset>.pkl
(top-10 chunks per system and query, reused by the QA evaluation).
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rag_core import (BM25, Dense, Normalized, char_recall, doc_from_pdf, doc_from_text,  # noqa: E402
                      encode_queries, flat_chunks, graph_expand, is_hit, rrf, structure_chunks)

from paths import ADNQA, W  # noqa: E402
KS = (1, 3, 5, 10)
SEED = 13


# ---------------------------------------------------------------------------
# Dataset adapters: (docs, queries) with queries = [{id, query, gold: [(doc, s, e)], scope}]
# scope None = whole corpus, else a doc id the query is restricted to
# ---------------------------------------------------------------------------


NUMLINE = re.compile(r"^(\d+(?:\.\d+)+)\b")


def adn_gold_spans(doc, refs: list[str]) -> list[tuple[str, int, int]]:
    """Locate a cited ADN paragraph independently of the decoder: from the first
    body line that starts with its number to the next numbered line that is not
    one of its sub-paragraphs."""
    out = []
    for ref in refs:
        start_line = None
        for i, ln in enumerate(doc.lines):
            if ln.page > 12 and re.match(re.escape(ref) + r"(\s|$)", ln.text) and not re.search(r"\.{4,}", ln.text):
                start_line = i
                break
        if start_line is None:
            continue
        end = len(doc.text)
        for ln, (s, _) in zip(doc.lines[start_line + 1:], doc.offsets[start_line + 1:]):
            m = NUMLINE.match(ln.text)
            if m and not (m.group(1) + ".").startswith(ref + ".") and m.group(1) != ref:
                end = s
                break
        out.append((doc.id, doc.offsets[start_line][0], end))
    return out


def load_adn(limit):
    doc = doc_from_pdf("adn", W / "raw/adn/adn2025_en.pdf")
    rows = [json.loads(l) for l in open(ADNQA / "retrieval.jsonl")]
    test = set(json.load(open(ADNQA / "splits.json"))["test"])
    queries, unlocated = [], 0
    for r in rows:
        gold = adn_gold_spans(doc, r["gold"])
        if not gold:
            unlocated += 1
            continue
        queries.append({"id": r["id"], "query": r["query"], "gold": gold, "scope": None,
                        "split": "test" if r["id"] in test else "dev"})
    print(f"adn: {len(queries)} queries ({unlocated} with no locatable gold)")
    return {"adn": doc}, queries[:limit] if limit else queries


OBLIQA_PDFS = {2: "CIB_VER04.030220", 5: "FP_VER01.110319", 11: "MKT_VER08.181223"}


def load_obliqa(limit):
    repo = W / "raw/obliqa/repo"
    docs = {}
    for did, name in OBLIQA_PDFS.items():
        p = W / f"raw/obliqa/pdf/{name}.pdf"
        if p.exists() and p.stat().st_size > 10000:
            docs[str(did)] = doc_from_pdf(str(did), p)
    norm = {d: Normalized(doc.text) for d, doc in docs.items()}
    rows = json.load(open(repo / "ObliQA_test.json"))
    queries, miss = [], 0
    for r in rows:
        if not all(str(p["DocumentID"]) in docs for p in r["Passages"]):
            continue
        gold = []
        for p in r["Passages"]:
            span = norm[str(p["DocumentID"])].locate(p["Passage"])
            if span:
                gold.append((str(p["DocumentID"]), *span))
        if not gold:
            miss += 1
            continue
        queries.append({"id": r["QuestionID"], "query": r["Question"], "gold": gold, "scope": None})
    print(f"obliqa: docs {sorted(docs)}; {len(queries)} queries ({miss} with unlocatable passages)")
    return docs, queries[:limit] if limit else queries


def load_lbrag(limit, per_task=194):
    base = W / "raw/lbrag/data"
    docs, queries = {}, []
    for task in ("contractnli", "cuad", "maud", "privacy_qa"):
        tests = json.load(open(base / f"benchmarks/{task}.json"))["tests"]
        rng = random.Random(SEED)
        tests = rng.sample(tests, min(per_task, len(tests)))  # LegalBench-RAG-mini size
        for k, t in enumerate(tests):
            for sn in t["snippets"]:
                if sn["file_path"] not in docs:
                    docs[sn["file_path"]] = doc_from_text(sn["file_path"], (base / "corpus" / sn["file_path"]).read_text())
            queries.append({"id": f"{task}-{k}", "query": t["query"], "task": task, "scope": None,
                            "gold": [(sn["file_path"], *sn["span"]) for sn in t["snippets"]]})
    print(f"lbrag: {len(docs)} docs, {len(queries)} queries")
    return docs, queries[:limit] if limit else queries


def load_qasper(limit):
    data = json.load(open(W / "raw/qasper/qasper-test-v0.3.json"))
    ids = (W / "raw/qasper/sample100.txt").read_text().split()
    docs, queries, miss = {}, [], 0
    for pid in ids:
        pdf = W / f"raw/qasper/pdf/{pid}.pdf"
        if not pdf.exists() or pdf.stat().st_size < 10000:
            continue
        try:
            docs[pid] = doc_from_pdf(pid, pdf)
        except Exception as e:  # noqa: BLE001
            print("skip", pid, e)
            continue
        norm = Normalized(docs[pid].text)
        for qa in data[pid]["qas"]:
            ev = {e for a in qa["answers"] for e in a["answer"]["evidence"] if not e.startswith("FLOAT SELECTED")}
            gold = [(pid, *s) for s in (norm.locate(e) for e in ev) if s]
            if not ev:
                continue  # unanswerable / table-only evidence
            if not gold:
                miss += 1
                continue
            queries.append({"id": qa["question_id"], "query": qa["question"], "gold": gold, "scope": pid,
                            "answers": qa["answers"]})
    print(f"qasper: {len(docs)} papers, {len(queries)} queries ({miss} with unlocatable evidence)")
    return docs, queries[:limit] if limit else queries


LOADERS = {"adn": load_adn, "obliqa": load_obliqa, "lbrag": load_lbrag, "qasper": load_qasper}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(name, chunks, queries, qvecs, cache, expand=False):
    texts = [c.text for c in chunks]
    t = time.time()
    bm = BM25(texts)
    dense = Dense(texts, cache)
    by_scope = defaultdict(list)
    for i, c in enumerate(chunks):
        by_scope[c.doc].append(i)
    results, hits_out = {}, {}
    for ret in ("bm25", "dense", "hybrid"):
        agg = defaultdict(float)
        hits_out[ret] = []
        for q, qv in zip(queries, qvecs):
            if q["scope"]:
                pool = by_scope[q["scope"]]
                sub_bm = q.setdefault("_bm", {}).get(name)
                if sub_bm is None:
                    sub_bm = BM25([texts[i] for i in pool])
                    q["_bm"][name] = sub_bm
                b = [pool[i] for i in sub_bm.search(q["query"], 50)]
                sc = dense.emb[pool] @ qv
                d = [pool[i] for i in sc.argsort()[::-1][:50]]
            else:
                b = bm.search(q["query"], 50)
                d = dense.search(q["query"], 50, qv)
            ranked = {"bm25": b, "dense": d, "hybrid": rrf(b, d, top=50)}[ret]
            if expand:
                ranked = graph_expand(ranked, chunks, 50)
            top = ranked[:10]
            hits_out[ret].append(top)
            first = next((r for r, i in enumerate(top) if any(is_hit(chunks[i], g) for g in q["gold"])), None)
            for k in KS:
                agg[f"hit@{k}"] += first is not None and first < k
                agg[f"crecall@{k}"] += char_recall([chunks[i] for i in top[:k]], q["gold"])
                agg[f"chars@{k}"] += sum(chunks[i].end - chunks[i].start for i in top[:k])
            agg["mrr@10"] += 1 / (first + 1) if first is not None else 0
        n = len(queries) or 1
        results[ret] = {k: round(v / n, 4) for k, v in agg.items()}
    print(f"  {name}: {len(chunks)} chunks, {time.time() - t:.0f}s  hybrid hit@5={results['hybrid']['hit@5']}")
    return results, hits_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=LOADERS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--backend", default="bge", choices=("bge", "openai"))
    ap.add_argument("--no-ablation", action="store_true", help="skip structure-noheader")
    args = ap.parse_args()
    import rag_core
    rag_core.BACKEND = args.backend
    docs, queries = LOADERS[args.dataset](args.limit)
    qvecs = encode_queries([q["query"] for q in queries])

    systems = {
        "flat": [c for d in docs.values() for c in flat_chunks(d)],
        "structure": [c for d in docs.values() for c in structure_chunks(d)],
    }
    if not args.no_ablation:
        systems["structure-noheader"] = [c for d in docs.values() for c in structure_chunks(d, header=False)]
    metrics, hits = {"n_queries": len(queries), "n_docs": len(docs)}, {}
    cache_dir = W / "runs/emb" / args.dataset
    for name, chunks in systems.items():
        metrics[name], hits[name] = evaluate(name, chunks, queries, qvecs, cache_dir / f"{name}.npy")
    metrics["structure+graph"], hits["structure+graph"] = evaluate(
        "structure+graph", systems["structure"], queries, qvecs, cache_dir / "structure.npy", expand=True)

    if args.dataset == "adn":  # also report the test split alone
        idx = [i for i, q in enumerate(queries) if q.get("split") == "test"]
        metrics["test_split_n"] = len(idx)
    for q in queries:
        q.pop("_bm", None)
    metrics["backend"] = args.backend
    out = W / f"runs/retrieval_{args.dataset}_{args.backend}.json"
    out.write_text(json.dumps(metrics, indent=1))
    with open(W / f"runs/hits_{args.dataset}_{args.backend}.pkl", "wb") as f:
        pickle.dump({"queries": queries, "systems": {n: [(c.doc, c.start, c.end, c.text, c.node) for c in ch]
                                                     for n, ch in systems.items()}, "hits": hits}, f)
    print(json.dumps({s: metrics[s]["hybrid"] for s in metrics if isinstance(metrics[s], dict)}, indent=1))


if __name__ == "__main__":
    main()
