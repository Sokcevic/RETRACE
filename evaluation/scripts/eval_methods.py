"""Retrieval under matched context budgets: vanilla RAG vs RETRACE variants.

Every system ranks its units and fills a context of B words in rank order (the
last unit is cut at the budget), as in DOS RAG's matched-token-budget protocol.
DOS RAG selects exactly the vanilla units and only re-orders them by document
position, so its retrieval scores equal vanilla's; it differs only at QA time.

Systems (local models only: BM25 + bge-small-en-v1.5)
  flat-N           fixed N-word windows (stride 0.8 N); flat-200 + dense = vanilla RAG
  struct-N         RETRACE: one chunk per decoded section with its heading path, split at N words
  struct-200-nohdr same without the heading path
  paragraph        RETRACE paragraph units: struct-200 pieces; a section of at most PARA_CAP words
                   is shown whole at its best rank, longer sections contribute their pieces
                   in rank order
Retrievers: dense, hybrid (BM25 + dense, RRF)
Metrics at B in BUDGETS words: hit (a selected unit covers a gold span, LegalBench-RAG
overlap rule), char recall of the gold text; plus hit@1 unit.

Usage: python eval_methods.py <adn|obliqa|lbrag|qasper> [--small]   (--small: only 200-word systems)
  -> runs/methods_<ds>.json, runs/methods_<ds>.pkl (per-query values)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rag_core  # noqa: E402
from rag_core import (BM25, Chunk, Dense, WORD_SPAN_RE, char_recall, encode_queries, flat_chunks,  # noqa: E402
                      is_hit, rrf, structure_chunks)
from eval_retrieval import LOADERS, W  # noqa: E402

BUDGETS = (500, 1000, 2000, 4000)
PARA_CAP = 400
POOL = 200
rag_core.BACKEND = "bge"


def with_size(n: int, fn, *a, **kw):
    old = rag_core.CHUNK_WORDS, rag_core.STRIDE
    rag_core.CHUNK_WORDS, rag_core.STRIDE = n, int(0.8 * n)
    try:
        return fn(*a, **kw)
    finally:
        rag_core.CHUNK_WORDS, rag_core.STRIDE = old


def trim(doc, c: Chunk, words: int) -> Chunk:
    """The first `words` words of a chunk's span (heading prefix not counted)."""
    spans = [m.end() for m in WORD_SPAN_RE.finditer(doc.text, c.start, c.end)]
    if len(spans) <= words:
        return c
    return Chunk(c.doc, c.start, spans[words - 1], c.text, c.node)


class System:
    def __init__(self, name: str, docs: dict, chunks: list[Chunk], paragraph: bool = False):
        self.name, self.docs, self.chunks, self.paragraph = name, docs, chunks, paragraph
        texts = [c.text for c in chunks]
        t = time.time()
        self.bm = BM25(texts)
        cache = name.replace("struct-200-nohdr", "structure-noheader").replace("struct-200", "structure") \
                    .replace("flat-200", "flat").replace("paragraph", "structure")
        self.dense = Dense(texts, W / f"runs/emb/{DS}/{cache}.npy")
        self.scoped_bm = {}
        self.by_doc = defaultdict(list)
        for i, c in enumerate(chunks):
            self.by_doc[c.doc].append(i)
        if paragraph:  # unit = (doc, node); its spans and word count
            self.units = defaultdict(list)
            for i, c in enumerate(chunks):
                self.units[(c.doc, c.node)].append(i)
        print(f"  {name}: {len(chunks)} chunks, index {time.time() - t:.0f}s", flush=True)

    def ranked(self, q, qv, ret: str) -> list[int]:
        if q["scope"]:
            pool = self.by_doc[q["scope"]]
            if ret == "hybrid" and q["scope"] not in self.scoped_bm:
                self.scoped_bm[q["scope"]] = BM25([self.chunks[i].text for i in pool])
            b = [pool[i] for i in self.scoped_bm[q["scope"]].search(q["query"], POOL)] if ret == "hybrid" else []
            d = [pool[i] for i in np.argsort(-(self.dense.emb[pool] @ qv))[:POOL]]
        else:
            b = self.bm.search(q["query"], POOL) if ret == "hybrid" else []
            d = list(np.argsort(-(self.dense.emb @ qv))[:POOL])
        return rrf(b, d, top=POOL) if ret == "hybrid" else d

    def context(self, ranked: list[int], budget: int, max_units: int | None = None) -> list[Chunk]:
        """Units in rank order until `budget` words (prefix words count too)."""
        out, used, seen = [], 0, set()
        for i in ranked:
            if used >= budget or (max_units and len(out) >= max_units):
                break
            c = self.chunks[i]
            if self.paragraph:
                key = (c.doc, c.node)
                members = [self.chunks[j] for j in self.units[key]]
                body = sum(len(self.docs[c.doc].text[m.start:m.end].split()) for m in members)
                if c.node != "front-matter" and body <= PARA_CAP:
                    # short section: shown whole, once; its pieces are contiguous
                    if key in seen:
                        continue
                    seen.add(key)
                    c = Chunk(c.doc, min(m.start for m in members), max(m.end for m in members), c.text, c.node)
                # long section / front matter: each retrieved piece on its own, in rank order
            prefix = len(c.text.split()) - len(self.docs[c.doc].text[c.start:c.end].split())
            room = budget - used - max(0, prefix)
            if room <= 0:
                break
            c = trim(self.docs[c.doc], c, room)
            out.append(c)
            used += max(0, prefix) + len(self.docs[c.doc].text[c.start:c.end].split())
        return out


DS = ""


def main():
    global DS
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=LOADERS)
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    DS = args.dataset
    docs, queries = LOADERS[DS](None)
    if DS == "adn":  # gold from the decoded structure (eval_paragraph.struct_gold)
        from eval_paragraph import DS as ADNQA, paragraph_units, struct_gold
        doc = docs["adn"]
        units = paragraph_units(doc, structure_chunks(doc))
        rows = {r["id"]: r for r in map(json.loads, open(ADNQA / "retrieval.jsonl"))}
        for q in queries:
            q["gold"] = struct_gold(doc, units, rows[q["id"]]["gold"])
    sizes = (200,) if args.small else (100, 200, 400)
    systems = []
    for n in sizes:
        systems.append(System(f"flat-{n}", docs, [c for d in docs.values() for c in with_size(n, flat_chunks, d)]))
        systems.append(System(f"struct-{n}", docs, [c for d in docs.values() for c in with_size(n, structure_chunks, d)]))
    s200 = next(s for s in systems if s.name == "struct-200")
    systems.append(System("paragraph", docs, s200.chunks, paragraph=True))
    if not args.small:
        systems.append(System("struct-200-nohdr", docs,
                              [c for d in docs.values() for c in structure_chunks(d, header=False)]))
    qvecs = encode_queries([q["query"] for q in queries])
    per = {}
    for s in systems:
        for ret in ("dense", "hybrid"):
            m = defaultdict(list)
            for q, qv in zip(queries, qvecs):
                ranked = s.ranked(q, qv, ret)
                for B in BUDGETS:
                    ctx = s.context(ranked, B)
                    m[f"hit@{B}w"].append(any(is_hit(c, g) for c in ctx for g in q["gold"]))
                    m[f"recall@{B}w"].append(char_recall(ctx, q["gold"]))
                first = s.context(ranked, 10 ** 7, max_units=1)
                m["hit@1unit"].append(any(is_hit(c, g) for c in first for g in q["gold"]))
            per[(s.name, ret)] = {k: np.array(v, float) for k, v in m.items()}
            print(f"  {s.name:17s} {ret:6s} " + " ".join(f"{k}={v.mean():.3f}" for k, v in per[(s.name, ret)].items()
                                                       if k.startswith("hit")), flush=True)
    report = {"parser": rag_core.PARSER, "n_queries": len(queries), "n_docs": len(docs),
              "results": {f"{n} [{r}]": {k: round(float(v.mean()), 4) for k, v in m.items()} for (n, r), m in per.items()}}
    (W / f"runs/methods_{DS}{rag_core.tag()}.json").write_text(json.dumps(report, indent=1))
    with open(W / f"runs/methods_{DS}{rag_core.tag()}.pkl", "wb") as f:
        pickle.dump({"ids": [q["id"] for q in queries], "per": per}, f)


if __name__ == "__main__":
    main()
