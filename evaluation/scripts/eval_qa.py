"""QA evaluation: does better indexing give better answers?

  adn     ADN-QA multiple choice (accuracy), contexts: closed-book, flat, structure,
          structure+graph, oracle (the cited ADN paragraphs)
  qasper  QASPER answers (official token F1), contexts: flat, structure,
          structure+graph, oracle (gold evidence)

Retrieval is hybrid (BM25 + bge-small, RRF), top-K chunks, same K for all systems.
LLM: any OpenAI-compatible endpoint (vLLM): RETRACE_LLM_URL (default http://localhost:8010/v1),
RETRACE_LLM_KEY (default "EMPTY"). Paper runs: Qwen3-32B served by vLLM, thinking off, temperature 0.
Usage: python eval_qa.py <adn|qasper> [--limit N] [--model qwen3-32b] [--split test]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rag_core import BM25, Dense, encode_queries, flat_chunks, graph_expand, rrf, structure_chunks  # noqa: E402
from eval_retrieval import load_adn, load_qasper, adn_gold_spans  # noqa: E402

from paths import ADNQA, W  # noqa: E402
K = 5
CACHE = W / "runs/llm_cache.jsonl"


def api_key() -> str:
    """Only for the optional OpenAI embedding backend of rag_core (not used in the paper runs)."""
    return os.environ.get("OPENAI_API_KEY", "")


class LLM:
    def __init__(self, model: str, base_url: str | None = None):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(base_url=base_url or os.environ.get("RETRACE_LLM_URL", "http://localhost:8010/v1"),
                             api_key=os.environ.get("RETRACE_LLM_KEY", "EMPTY"), timeout=600)
        # Qwen3: answer directly, no reasoning trace
        self.extra = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3" in model.lower() else {}
        self.cache = {}
        if CACHE.exists():
            for line in CACHE.open():
                r = json.loads(line)
                self.cache[r["k"]] = r["v"]
        self.out = CACHE.open("a")

    def __call__(self, system: str, user: str, max_tokens: int = 300) -> str:
        k = hashlib.sha256(f"{self.model}\n{system}\n{user}".encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0, max_tokens=max_tokens, extra_body=self.extra or None,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                v = r.choices[0].message.content or ""
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 4:
                    raise
                import time
                time.sleep(5 * (attempt + 1))
        self.cache[k] = v
        self.out.write(json.dumps({"k": k, "v": v}) + "\n")
        self.out.flush()
        return v


def contexts_for(docs, queries, query_texts, dataset):
    """{system: [list of context strings per query]}"""
    qvecs = encode_queries(query_texts)
    out = {}
    for name in ("flat", "structure"):
        chunks = [c for d in docs.values() for c in (flat_chunks(d) if name == "flat" else structure_chunks(d))]
        texts = [c.text for c in chunks]
        dense = Dense(texts, W / f"runs/emb/{dataset}/{name}.npy")
        by_doc = defaultdict(list)
        for i, c in enumerate(chunks):
            by_doc[c.doc].append(i)
        bm_all = BM25(texts) if dataset == "adn" else None
        sel, sel_graph = [], []
        for q, qt, qv in zip(queries, query_texts, qvecs):
            if q["scope"]:
                pool = by_doc[q["scope"]]
                bm = BM25([texts[i] for i in pool])
                b = [pool[i] for i in bm.search(qt, 50)]
                d = [pool[i] for i in (dense.emb[pool] @ qv).argsort()[::-1][:50]]
            else:
                b = bm_all.search(qt, 50)
                d = dense.search(qt, 50, qv)
            ranked = rrf(b, d, top=50)
            sel.append([chunks[i].text for i in ranked[:K]])
            if name == "structure":
                sel_graph.append([chunks[i].text for i in graph_expand(ranked, chunks, K)])
        out[name] = sel
        if name == "structure":
            out["structure+graph"] = sel_graph
    out["oracle"] = [[docs[g[0]].text[g[1]:g[2]][:6000] for g in q["gold"]] for q in queries]
    return out


MCQ_SYSTEM = ("You are an expert on the ADN (European Agreement concerning the International Carriage of "
              "Dangerous Goods by Inland Waterways). Answer the multiple-choice question. Reply with the "
              "letter of the correct option only.")


def fmt_mcq(q, ctx):
    opts = "\n".join(f"{k} {v}" for k, v in q["options"].items())
    c = ("Excerpts from the ADN 2025 regulations:\n\n" + "\n\n---\n\n".join(ctx) + "\n\n") if ctx else ""
    return f"{c}Question: {q['question']}\n{opts}\nAnswer:"


def run_adn(args, llm):
    docs, rqueries = load_adn(None)
    doc = docs["adn"]
    qa = [json.loads(l) for l in open(ADNQA / "qa.jsonl")]
    split = set(json.load(open(ADNQA / "splits.json"))[args.split])
    qa = [q for q in qa if q["id"] in split][: args.limit or None]
    queries = [{"id": q["id"], "scope": None, "gold": adn_gold_spans(doc, q["adn_refs"])} for q in qa]
    qtexts = [q["question"] + " " + " ".join(q["options"].values()) for q in qa]
    ctx = contexts_for(docs, queries, qtexts, "adn")
    ctx["closed-book"] = [[] for _ in qa]
    results, items = {}, {}
    for system, cs in ctx.items():
        def one(i):
            if system == "oracle" and not queries[i]["gold"]:
                return None
            a = llm(MCQ_SYSTEM, fmt_mcq(qa[i], cs[i]), max_tokens=5)
            m = re.search(r"\b([ABCD])\b", a.strip().upper())
            return (m.group(1) if m else "?") == qa[i]["answer"]
        with cf.ThreadPoolExecutor(8) as ex:
            ok = list(ex.map(one, range(len(qa))))
        cited = [o for o, q in zip(ok, qa) if q["adn_refs"] and o is not None]
        general = [o for o, q in zip(ok, qa) if not q["adn_refs"] and o is not None]
        allv = [o for o in ok if o is not None]
        items[system] = [None if o is None else float(o) for o in ok]
        results[system] = {"acc": round(sum(allv) / max(1, len(allv)), 4), "n": len(allv),
                           "acc_cited": round(sum(cited) / max(1, len(cited)), 4), "n_cited": len(cited),
                           "acc_basic": round(sum(general) / max(1, len(general)), 4), "n_basic": len(general)}
        print(system, results[system])
    save_items("adn", args.model, "accuracy", [q["id"] for q in qa], items)
    return results


def save_items(dataset, model, metric, ids, items):
    (W / f"runs/qa_items_{dataset}_{model}.json").write_text(
        json.dumps({"metric": metric, "ids": ids, "items": items}))


QASPER_SYSTEM = ("Answer the question about the scientific paper using only the excerpts. Be concise: "
                 "give a short phrase, 'Yes' or 'No', or reply 'Unanswerable' if the excerpts do not "
                 "contain the answer.")


def run_qasper(args, llm):
    sys.path.insert(0, str(W / "raw/qasper"))
    from qasper_evaluator import token_f1_score  # official answer F1
    docs, queries = load_qasper(None)
    queries = queries[: args.limit or None]
    ctx = contexts_for(docs, queries, [q["query"] for q in queries], "qasper")
    results, items = {}, {}

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

    for system, cs in ctx.items():
        def one(i):
            q = queries[i]
            user = "Excerpts:\n\n" + "\n\n---\n\n".join(cs[i]) + f"\n\nQuestion: {q['query']}\nAnswer:"
            pred = llm(QASPER_SYSTEM, user, max_tokens=120).strip()
            return max(token_f1_score(pred, g) for g in golds(q))
        with cf.ThreadPoolExecutor(8) as ex:
            f1 = list(ex.map(one, range(len(queries))))
        results[system] = {"answer_f1": round(sum(f1) / max(1, len(f1)), 4), "n": len(f1)}
        items[system] = [float(x) for x in f1]
        print(system, results[system])
    save_items("qasper", args.model, "answer_f1", [q["id"] for q in queries], items)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=("adn", "qasper"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--split", default="test")
    ap.add_argument("--backend", default="bge", choices=("bge", "openai"))
    args = ap.parse_args()
    import rag_core
    rag_core.BACKEND = args.backend
    llm = LLM(args.model)
    res = run_adn(args, llm) if args.dataset == "adn" else run_qasper(args, llm)
    out = W / f"runs/qa_{args.dataset}_{args.model}.json"
    out.write_text(json.dumps({"model": args.model, "k": K, "results": res}, indent=1))


if __name__ == "__main__":
    main()
