"""Paragraph-level evaluation on ADN-QA: the index unit is one decoded ADN
paragraph (heading up to the next heading), never cut.

Paragraphs longer than CHUNK_WORDS are indexed as several pieces (the same
pieces as the `structure` system, so embeddings are reused), but retrieval
returns each paragraph once: pieces are ranked, then deduplicated by paragraph ID.

Retrieval is scored on IDs: a retrieved paragraph p answers a cited paragraph g
when p == g or p lies below g (e.g. 8.1.2.1 for a citation of 8.1.2).
The flat baseline cannot be scored on IDs; it is scored by character overlap
with the gold paragraph, as in eval_retrieval.py.

Paragraphs longer than PARA_CAP words (tables, definition lists) are shown as
their best-matching piece. QA gives the model top-5 paragraphs, and paragraphs
in rank order up to the flat system's word budget (5 x 200 words).

Usage: python eval_paragraph.py [--backend openai|bge] [--qa]
  -> runs/paragraph_adn_<backend>.json, runs/qa_items_adn-paragraph_<model>.json
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
import rag_core  # noqa: E402
from rag_core import BM25, Chunk, encode_queries, flat_chunks, is_hit, rrf, structure_chunks  # noqa: E402
from eval_retrieval import adn_gold_spans, load_adn  # noqa: E402
from significance import bootstrap  # noqa: E402

from paths import W  # noqa: E402
from paths import ADNQA as DS  # noqa: E402
KS = (1, 3, 5, 10)
POOL = 300  # pieces taken from each retriever before deduplication
PARA_CAP = 400  # longer paragraphs (tables, definition lists) are shown as their best-matching piece
BUDGET = 1000  # words, the flat system's context size (5 x 200)


def load_emb(texts: list[str], name: str) -> np.ndarray:
    """Reuse the cached piece embeddings of eval_retrieval.py (bge caches predate the backend tag)."""
    import hashlib
    key = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
    d = W / "runs/emb/adn"
    tags = [f"{name}-openai-{key}"] if rag_core.BACKEND == "openai" else [f"{name}-bge-{key}", f"{name}-{key}"]
    for t in tags:
        if (d / f"{t}.npy").exists():
            return np.load(d / f"{t}.npy")
    return rag_core.Dense(texts, d / f"{name}.npy").emb


class Index:
    def __init__(self, chunks: list[Chunk], name: str, ids: bool):
        self.chunks, self.ids = chunks, ids
        texts = [c.text for c in chunks]
        self.bm = BM25(texts)
        self.emb = load_emb(texts, name)

    def search(self, query: str, qvec: np.ndarray, ret: str = "hybrid") -> list:
        """Ranked paragraph IDs (ids=True) or chunk indices (flat)."""
        b = self.bm.search(query, POOL)
        d = list(np.argsort(-(self.emb @ qvec))[:POOL])
        ranked = {"bm25": b, "dense": d, "hybrid": rrf(b, d, top=POOL)}[ret]
        if not self.ids:
            return ranked
        out, self.best = [], {}
        for i in ranked:
            n = self.chunks[i].node
            if n not in self.best:
                out.append(n)
                self.best[n] = i
        return out


def answers(p: str | None, g: str) -> bool:
    return p is not None and (p == g or p.startswith(g + "."))


def paragraph_units(doc, chunks: list[Chunk]) -> dict[str, dict]:
    """node -> {prefix, spans}; a node can own several disjoint spans."""
    units: dict[str, dict] = {}
    for c in chunks:
        u = units.setdefault(c.node, {"spans": [], "prefix": ""})
        body = doc.text[c.start:c.end].strip()
        if not u["prefix"]:
            u["prefix"] = c.text[: max(0, len(c.text) - len(body))].strip()
        if u["spans"] and c.start <= u["spans"][-1][1]:
            u["spans"][-1][1] = max(u["spans"][-1][1], c.end)
        else:
            u["spans"].append([c.start, c.end])
    return units


def unit_text(doc, u: dict, cap: int) -> str:
    words = " ".join(doc.text[s:e] for s, e in u["spans"]).split()
    body = " ".join(words[:cap]) + (" [...]" if len(words) > cap else "")
    return (u["prefix"] + "\n" + body).strip()


def unit_words(doc, u: dict) -> int:
    return sum(len(doc.text[s:e].split()) for s, e in u["spans"])


def view(doc, para: "Index", units, p: str) -> str:
    """What the reader gets for paragraph p: all of it, or its best piece if it is long."""
    u = units[p]
    if unit_words(doc, u) <= PARA_CAP:
        return unit_text(doc, u, PARA_CAP)
    return para.chunks[para.best[p]].text


def view_chunks(doc, para: "Index", units, p: str) -> list[Chunk]:
    """The text spans behind view(): every span of the paragraph, or its best piece."""
    u = units[p]
    if unit_words(doc, u) <= PARA_CAP:
        return [Chunk(doc.id, s, e, "") for s, e in u["spans"]]
    return [para.chunks[para.best[p]]]


def struct_gold(doc, units, refs: list[str]) -> list[tuple[str, int, int]]:
    """Gold text of each cited paragraph: its decoded span plus those of its sub-paragraphs.
    Falls back to the line-based locator of eval_retrieval.py for refs the decoder missed;
    that locator can land on a paragraph number quoted in a table, so it is only a fallback."""
    out = []
    for g in refs:
        spans = sorted(tuple(sp) for p, u in units.items() if p and answers(p, g) for sp in u["spans"])
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out += [(doc.id, a, b) for a, b in merged] if merged else adn_gold_spans(doc, [g])
    return out


def within_budget(doc, para, units, ranked: list[str], budget: int) -> list[str]:
    """Paragraphs taken in rank order until the word budget is used up."""
    out, used = [], 0
    for p in ranked:
        if used >= budget:
            break
        out.append(p)
        used += len(view(doc, para, units, p).split())
    return out


def retrieval(doc, queries, rows, para: Index, flat: Index, units) -> tuple[dict, dict]:
    qvecs = encode_queries([q["query"] for q in queries])
    in_struct = [all(r["gold_in_structure"]) for r in rows]
    per = {s: {f"hit@{k}": [] for k in KS} | {"mrr@10": [], "all@5": [], "words@5": []}
           for s in ("flat", "paragraph", "paragraph-exact", "paragraph-text")}
    for q, r, qv in zip(queries, rows, qvecs):
        top = flat.search(q["query"], qv)[:10]
        for budget in (1000, 2000):  # flat chunks are 200 words
            per["flat"].setdefault(f"hit@{budget}w", []).append(
                any(is_hit(flat.chunks[i], g) for i in top[:budget // 200] for g in q["gold"]))
        first = next((k for k, i in enumerate(top) if any(is_hit(flat.chunks[i], g) for g in q["gold"])), None)
        _add(per["flat"], first, len(q["gold"]) and all(any(is_hit(flat.chunks[i], g) for i in top[:5]) for g in q["gold"]),
             sum(len(flat.chunks[i].text.split()) for i in top[:5]))
        ptop = para.search(q["query"], qv)[:30]
        ptop10 = ptop[:10]
        # paragraph / paragraph-exact: scored on IDs; paragraph-text: scored on text
        # overlap with the gold span, exactly like flat (checks the ID scoring is not easier)
        text_hit = lambda p, g: any(is_hit(c, g) for c in view_chunks(doc, para, units, p))
        for name, match, golds in (("paragraph", answers, r["gold"]), ("paragraph-exact", lambda p, g: p == g, r["gold"]),
                                   ("paragraph-text", text_hit, q["gold"])):
            first = next((k for k, p in enumerate(ptop10) if any(match(p, g) for g in golds)), None)
            _add(per[name], first, len(golds) and all(any(match(p, g) for p in ptop[:5]) for g in golds),
                 sum(len(view(doc, para, units, p).split()) for p in ptop[:5]))
            for budget in (1000, 2000):
                sel = within_budget(doc, para, units, ptop, budget)
                per[name].setdefault(f"hit@{budget}w", []).append(any(match(p, g) for p in sel for g in golds))
    report = {}
    for subset, mask in (("all", [True] * len(rows)), ("gold_in_structure", in_struct)):
        report[subset] = {"n": int(sum(mask))}
        for s, m in per.items():
            report[subset][s] = {k: round(float(np.mean([v for v, keep in zip(vals, mask) if keep])), 4)
                                 for k, vals in m.items()}
        report[subset]["paragraph vs flat"] = {
            k: bootstrap(np.array([v for v, keep in zip(per["paragraph"][k], mask) if keep], float),
                         np.array([v for v, keep in zip(per["flat"][k], mask) if keep], float), 10000)
            for k in ("hit@1", "hit@5", "mrr@10", "hit@1000w", "hit@2000w")}
    return report, per


def _add(m, first, all5, words):
    for k in KS:
        m[f"hit@{k}"].append(first is not None and first < k)
    m["mrr@10"].append(1 / (first + 1) if first is not None else 0.0)
    m["all@5"].append(bool(all5))
    m["words@5"].append(words)


def qa(doc, para: Index, units, model: str):
    from eval_qa import LLM, MCQ_SYSTEM, fmt_mcq
    llm = LLM(model)
    rows = [json.loads(l) for l in open(DS / "qa.jsonl")]
    split = set(json.load(open(DS / "splits.json"))["test"])
    rows = [q for q in rows if q["id"] in split]
    qtexts = [q["question"] + " " + " ".join(q["options"].values()) for q in rows]
    qvecs = encode_queries(qtexts)
    ctx = {"paragraph-top5": [], "paragraph-budget": []}
    for qt, qv in zip(qtexts, qvecs):
        ranked = para.search(qt, qv)
        ctx["paragraph-top5"].append([view(doc, para, units, p) for p in ranked[:5]])
        ctx["paragraph-budget"].append([view(doc, para, units, p) for p in within_budget(doc, para, units, ranked, BUDGET)])
    items, results = {}, {}
    for system, cs in ctx.items():
        def one(i):
            a = llm(MCQ_SYSTEM, fmt_mcq(rows[i], cs[i]), max_tokens=5)
            m = re.search(r"\b([ABCD])\b", a.strip().upper())
            return float((m.group(1) if m else "?") == rows[i]["answer"])
        with cf.ThreadPoolExecutor(8) as ex:
            ok = list(ex.map(one, range(len(rows))))
        items[system] = ok
        cited = [o for o, q in zip(ok, rows) if q["adn_refs"]]
        basic = [o for o, q in zip(ok, rows) if not q["adn_refs"]]
        results[system] = {"acc": round(np.mean(ok), 4), "acc_cited": round(np.mean(cited), 4),
                           "acc_basic": round(np.mean(basic), 4), "n": len(ok),
                           "context_words": round(float(np.mean([sum(len(t.split()) for t in c) for c in cs])))}
        print(system, results[system])
    # compare with the chunk-level systems of eval_qa.py (same questions, same order)
    old = json.load(open(W / f"runs/qa_items_adn_{model}.json"))
    pos = {i: k for k, i in enumerate(old["ids"])}
    idx = [pos[q["id"]] for q in rows]
    for base in ("flat", "structure"):
        b = np.array([old["items"][base][k] for k in idx], float)
        for system in ctx:
            results[f"{system} vs {base}"] = bootstrap(np.array(items[system], float), b, 10000)
    (W / f"runs/qa_items_adn-paragraph_{model}.json").write_text(
        json.dumps({"metric": "accuracy", "ids": [q["id"] for q in rows], "items": items}))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openai", choices=("openai", "bge"))
    ap.add_argument("--qa", action="store_true")
    ap.add_argument("--model", default="gpt-4.1-mini")
    args = ap.parse_args()
    rag_core.BACKEND = args.backend
    docs, queries = load_adn(None)
    doc = docs["adn"]
    rows = {r["id"]: r for r in map(json.loads, open(DS / "retrieval.jsonl"))}
    pchunks = structure_chunks(doc)
    units = paragraph_units(doc, pchunks)
    lens = [unit_words(doc, u) for u in units.values()]
    print(f"{len(units)} paragraphs, {len(pchunks)} pieces; words median {int(np.median(lens))}, "
          f"p90 {int(np.percentile(lens, 90))}, max {max(lens)}")
    para = Index(pchunks, "structure", ids=True)
    flat = Index(flat_chunks(doc), "flat", ids=False)
    allrows = list(rows.values())
    qs = [{"id": r["id"], "query": r["query"], "gold": struct_gold(doc, units, r["gold"])} for r in allrows]
    report, _ = retrieval(doc, qs, allrows, para, flat, units)
    report["paragraph_words"] = {"median": int(np.median(lens)), "p90": int(np.percentile(lens, 90))}
    if args.qa:
        report["qa"] = qa(doc, para, units, args.model)
    out = W / f"runs/paragraph_adn_{args.backend}.json"
    out.write_text(json.dumps(report, indent=1))
    for subset in ("all", "gold_in_structure"):
        print(f"\n== {subset} (n={report[subset]['n']})")
        for s in ("flat", "paragraph", "paragraph-exact", "paragraph-text"):
            print(f"  {s:16s}", {k: v for k, v in report[subset][s].items()})
        for k, v in report[subset]["paragraph vs flat"].items():
            print(f"  paragraph vs flat {k}: {v}")
    if args.qa:
        print(json.dumps(report["qa"], indent=1))


if __name__ == "__main__":
    main()
