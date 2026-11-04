"""Shared pieces of the RETRACE retrieval/QA evaluation.

A document is a text T plus its layout lines; every chunk and every gold
passage is a character span [start, end) of T, so chunkers are compared on
the same footing regardless of how they cut the text (LegalBench-RAG style).

Chunkers
  flat        fixed windows of CHUNK_WORDS words, STRIDE words apart
  structure   one chunk per decoded heading (RETRACE structure decoder), text up
              to the next heading, prefixed with the ancestor titles; longer
              chunks are split into CHUNK_WORDS pieces that keep the prefix
Retrievers
  bm25, dense (bge-small-en-v1.5), hybrid (reciprocal rank fusion)
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import paths  # noqa: F401  (puts the repo root on sys.path)
from regulatory_testgen.parsing.structure_decoder import Line, decode_structure, pdf_layout_lines  # noqa: E402
from regulatory_testgen.structure_ids import parent_of  # noqa: E402

CHUNK_WORDS = 200
STRIDE = 160
TOKEN_RE = re.compile(r"\w+")
WORD_SPAN_RE = re.compile(r"\S+")


@dataclass
class Chunk:
    doc: str
    start: int
    end: int
    text: str  # what is indexed (may carry a header prefix)
    node: str | None = None  # structure node id
    parent: str | None = None
    refs: list[str] = field(default_factory=list)


@dataclass
class Doc:
    id: str
    text: str
    lines: list[Line]
    offsets: list[tuple[int, int]]  # char span of each line in text


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


# PDF parser feeding the decoder: "pymupdf" (baseline clustering) or "mineru" (MinerU reading
# order, needs MinerU output under $RETRACE_WORK/raw/mineru/<dataset>/); env RETRACE_PARSER.
PARSER = os.environ.get("RETRACE_PARSER", "pymupdf")


def tag() -> str:
    """Suffix for output files: empty for the default parser."""
    return "" if PARSER == "pymupdf" else f"_{PARSER}"


def doc_from_pdf(doc_id: str, pdf: Path) -> Doc:
    lines = None
    if PARSER == "mineru":
        from paths import W
        from regulatory_testgen.parsing.mineru_lines import find_middle_json, mineru_layout_lines
        mj = next((m for d in sorted((W / "raw/mineru").iterdir()) if d.is_dir()
                   for m in [find_middle_json(d, Path(pdf).stem)] if m), None)
        if mj is None:
            print(f"no MinerU output for {pdf.name}; falling back to PyMuPDF")
        else:
            lines = mineru_layout_lines(pdf, mj)
    if lines is None:
        lines = pdf_layout_lines(pdf)
    offsets, pos = [], 0
    for ln in lines:
        offsets.append((pos, pos + len(ln.text)))
        pos += len(ln.text) + 1
    return Doc(doc_id, "\n".join(ln.text for ln in lines), lines, offsets)


def doc_from_text(doc_id: str, text: str) -> Doc:
    """Plain-text document (no fonts): every line is body size, nothing bold."""
    lines, offsets = [], []
    for m in re.finditer(r"[^\n]+", text):
        s = m.group(0)
        stripped = s.strip()
        if not stripped:
            continue
        lead = len(s) - len(s.lstrip())
        lines.append(Line(page=1, text=" ".join(stripped.split()), size=10.0, bold=False,
                          top=float(len(lines)), x0=float(lead)))
        offsets.append((m.start() + lead, m.start() + lead + len(stripped)))
    return Doc(doc_id, text, lines, offsets)


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------


def _split_words(doc: Doc, start: int, end: int, prefix: str, **meta) -> list[Chunk]:
    words = [(m.start() + start, m.end() + start) for m in WORD_SPAN_RE.finditer(doc.text[start:end])]
    if not words:
        return [Chunk(doc.id, start, end, prefix.strip(), **meta)] if prefix.strip() else []
    out = []
    for i in range(0, len(words), STRIDE if len(words) > CHUNK_WORDS else CHUNK_WORDS):
        piece = words[i:i + CHUNK_WORDS]
        s, e = piece[0][0], piece[-1][1]
        out.append(Chunk(doc.id, s, e, (prefix + doc.text[s:e]).strip(), **meta))
        if i + CHUNK_WORDS >= len(words):
            break
    return out


def flat_chunks(doc: Doc) -> list[Chunk]:
    return _split_words(doc, 0, len(doc.text), "")


REF_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+){1,6})(?![\d.]*\d)")


def structure_chunks(doc: Doc, start_re: str | None = None, *, header: bool = True) -> list[Chunk]:
    kwargs = {"start": start_re} if start_re else {}
    heads = sorted(decode_structure(doc.lines, **kwargs), key=lambda h: h.line_index)
    heads = [h for h in heads if h.kind != "container" or True]
    titles: dict[str, str] = {}
    norm = lambda cid: re.sub(r"^Volume [IVX]+ / ", "", cid)
    for h in heads:
        titles.setdefault(norm(h.clause_id), h.text[:120])
    ids = set(titles)
    chunks: list[Chunk] = []
    first = heads[0].line_index if heads else len(doc.lines)
    if first > 0:  # front matter before the first heading
        chunks += _split_words(doc, 0, doc.offsets[first - 1][1], "", node="front-matter")
    for k, h in enumerate(heads):
        s = doc.offsets[h.line_index][0]
        e = doc.offsets[heads[k + 1].line_index][0] if k + 1 < len(heads) else len(doc.text)
        if e <= s:
            continue
        cid = norm(h.clause_id)
        path, p = [], parent_of(cid)
        while p:
            if p in titles:
                path.append(f"{p} {titles[p]}".strip())
            p = parent_of(p)
        prefix = (" > ".join(reversed(path)) + "\n") if header and path else ""
        par = parent_of(cid)
        while par and par not in ids:
            par = parent_of(par)
        body = doc.text[s:e]
        refs = [r for r in dict.fromkeys(REF_RE.findall(body)) if r in ids and r != cid]
        chunks += _split_words(doc, s, e, prefix, node=cid, parent=par, refs=refs)
    return chunks


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------


def tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


class BM25:
    def __init__(self, texts: list[str]):
        from rank_bm25 import BM25Okapi
        self.bm = BM25Okapi([tokenize(t) or ["_"] for t in texts])

    def search(self, query: str, k: int) -> list[int]:
        scores = self.bm.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:k])


_MODEL = None
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BACKEND = "bge"  # or "openai" (text-embedding-3-small), set by the caller


def _openai_embed(texts: list[str]) -> np.ndarray:
    from openai import OpenAI
    from eval_qa import api_key
    client = OpenAI(api_key=api_key())
    out = []
    for i in range(0, len(texts), 256):
        batch = [t[:8000] or "_" for t in texts[i:i + 256]]
        for attempt in range(8):
            try:
                r = client.embeddings.create(model="text-embedding-3-small", input=batch)
                break
            except Exception:  # noqa: BLE001  (rate limit: 1M tokens/min)
                if attempt == 7:
                    raise
                import time
                time.sleep(min(60, 4 * 2 ** attempt))
        out += [d.embedding for d in r.data]
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def embedder():
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(16)
        _MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        _MODEL.max_seq_length = 320  # chunks are <= 200 words
    return _MODEL


class Dense:
    def __init__(self, texts: list[str], cache: Path | None = None):
        key = None
        if cache:
            import hashlib
            key = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
            cache = cache.with_name(f"{cache.stem}-{BACKEND}-{key}.npy")  # invalidated when the chunks change
        if cache and cache.exists():
            self.emb = np.load(cache)
        else:
            if BACKEND == "openai":
                self.emb = _openai_embed(texts)
            else:
                self.emb = embedder().encode(texts, batch_size=64, normalize_embeddings=True,
                                             show_progress_bar=False, convert_to_numpy=True)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.emb)

    def search(self, query: str, k: int, qvec: np.ndarray | None = None) -> list[int]:
        if qvec is None:
            qvec = embedder().encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = self.emb @ qvec
        return list(np.argsort(-scores)[:k])


def rrf(*rankings: list[int], k: int = 60, top: int = 10) -> list[int]:
    score: dict[int, float] = {}
    for r in rankings:
        for rank, i in enumerate(r):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=lambda i: -score[i])[:top]


def encode_queries(queries: list[str]) -> np.ndarray:
    if BACKEND == "openai":
        return _openai_embed(queries)
    return embedder().encode([QUERY_PREFIX + q for q in queries], batch_size=64,
                             normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# Graph expansion (structure + graph)
# ---------------------------------------------------------------------------


def graph_expand(hits: list[int], chunks: list[Chunk], budget: int) -> list[int]:
    """Add, after each retrieved structure chunk, the first chunk of its parent and
    of the nodes it cross-references, until `budget` chunks are selected."""
    first: dict[tuple[str, str], int] = {}
    for i, c in enumerate(chunks):
        if c.node:
            first.setdefault((c.doc, c.node), i)
    out: list[int] = []
    for i in hits:
        for j in [i] + [first.get((chunks[i].doc, n)) for n in [chunks[i].parent, *chunks[i].refs[:2]]]:
            if j is not None and j not in out:
                out.append(j)
            if len(out) >= budget:
                return out
    return out


# ---------------------------------------------------------------------------
# Gold matching
# ---------------------------------------------------------------------------


def overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def is_hit(chunk: Chunk, gold: tuple[str, int, int]) -> bool:
    if chunk.doc != gold[0]:
        return False
    ov = overlap((chunk.start, chunk.end), (gold[1], gold[2]))
    return ov > 0 and ov >= 0.5 * min(gold[2] - gold[1], chunk.end - chunk.start)


def char_recall(chunks: list[Chunk], golds: list[tuple[str, int, int]]) -> float:
    total = sum(g[2] - g[1] for g in golds) or 1
    covered = 0
    for g in golds:
        spans = sorted((max(c.start, g[1]), min(c.end, g[2])) for c in chunks if c.doc == g[0]
                       and overlap((c.start, c.end), (g[1], g[2])))
        cur = g[1]
        for s, e in spans:
            s = max(s, cur)
            if e > s:
                covered += e - s
                cur = e
    return covered / total


class Normalized:
    """Lower-cased alphanumeric view of a text with a map back to original offsets,
    for locating gold passages whose whitespace/hyphenation differs from T."""

    def __init__(self, text: str):
        keep = [(i, ch.lower()) for i, ch in enumerate(text) if ch.isalnum()]
        self.idx = [i for i, _ in keep]
        self.s = "".join(c for _, c in keep)

    def locate(self, passage: str, anchor: int = 40) -> tuple[int, int] | None:
        p = "".join(ch.lower() for ch in passage if ch.isalnum())
        if len(p) < 12:
            return None
        if len(p) <= anchor:
            i = self.s.find(p)
            return (self.idx[i], self.idx[i + len(p) - 1] + 1) if i >= 0 else None
        starts = [o for o in range(0, len(p) - anchor + 1, max(1, (len(p) - anchor) // 4 or 1))][:5]
        found = []
        for o in starts:
            i = self.s.find(p[o:o + anchor])
            if i >= 0:
                found.append((o, i))
        if not found:
            return None
        # anchor the span on the found pieces, extrapolating to the passage length
        o, i = found[0]
        s = max(0, i - o)
        o2, i2 = found[-1]
        e = min(len(self.s) - 1, i2 + (len(p) - o2) - 1)
        if e <= s or e - s > 3 * len(p):
            e = min(len(self.s) - 1, s + len(p) - 1)
        return self.idx[s], self.idx[e] + 1
