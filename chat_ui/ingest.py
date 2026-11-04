"""
ingest.py — Turn an uploaded document into a graph the chat UI can query.

Pipeline: (PDF/DOCX/PPTX/XLSX/image) --MinerU--> Markdown --parse--> Clauses
    --LLM classify--> categories --extract tables--> RegulationTable
    --RegulatoryGraph.build--> graph

MinerU runs as a subprocess from its own venv (chat_ui/venv_mineru) because it
needs Python 3.10-3.13 and a large, separate dependency set (torch, etc.) that
would conflict with the pipeline's venv. Plain Markdown files skip MinerU
entirely and are parsed directly.

Offloading to a remote GPU: set MINERU_API_URL to a `mineru-api` server's base
URL (e.g. "http://gpu-box:8008") and the local `mineru` CLI acts as a thin
client — it uploads the file and downloads the result, with all model
inference (layout/OCR/VLM) running on that remote server instead of the local
CPU. See chat_ui/README.md for how to start `mineru-api` on the GPU box.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
MINERU_BIN = _ROOT / "chat_ui" / "venv_mineru" / "bin" / "mineru"
UPLOAD_DIR = _ROOT / "chat_ui" / "uploads"
MINERU_API_URL = os.environ.get("MINERU_API_URL")  # e.g. "http://gpu-box:8008" — None = run locally
MINERU_BACKEND = os.environ.get("MINERU_BACKEND", "pipeline")  # "pipeline" (CPU-friendly) or
                                                                 # "hybrid-engine"/"vlm-engine" (GPU, higher throughput+accuracy)

MARKDOWN_SUFFIXES = {".md", ".markdown"}
MINERU_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg"}
SUPPORTED_SUFFIXES = MARKDOWN_SUFFIXES | MINERU_SUFFIXES

ProgressCB = Callable[[float, str], None]


class IngestError(RuntimeError):
    pass


def _noop(_frac: float, _msg: str) -> None:
    pass


def _ensure_pipeline_on_path() -> None:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


class ConversionResult:
    """Output of convert_to_markdown: the parseable markdown plus (when available)
    the artifacts needed to trace clauses back to the source PDF."""

    def __init__(self, md_path: Path, middle_json_path: Path | None, source_pdf_path: Path | None):
        self.md_path = md_path
        self.middle_json_path = middle_json_path
        self.source_pdf_path = source_pdf_path


def convert_to_markdown(input_path: Path, work_dir: Path, progress_cb: ProgressCB = _noop) -> ConversionResult:
    """Turn input_path into a markdown file (plus source-tracing artifacts).

    Markdown input is returned unchanged, with no source-tracing info (no PDF to
    highlight). Everything else is run through MinerU's CPU-only "pipeline"
    backend, which also emits a `<stem>_middle.json` with per-block page/bbox
    data used later to trace clauses back to their location in the PDF.
    """
    if input_path.suffix.lower() in MARKDOWN_SUFFIXES:
        return ConversionResult(input_path, None, None)

    if input_path.suffix.lower() not in MINERU_SUFFIXES:
        raise IngestError(
            f"Unsupported file type '{input_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    # Cache: MinerU output is written into work_dir, which the caller keys by the
    # file's content hash (uploads/<sha256>/output/mineru_out). So if a previous
    # run already produced markdown here for the same file, reuse it and skip the
    # (slow) conversion entirely.
    cached = _locate_mineru_output(work_dir, input_path)
    if cached is not None:
        progress_cb(0.38, f"Using cached MinerU conversion for {input_path.name}.")
        return cached

    if not MINERU_BIN.exists():
        raise IngestError(
            f"MinerU is not installed at {MINERU_BIN}. Set it up with:\n"
            f"  python3.12 -m venv chat_ui/venv_mineru\n"
            f"  chat_ui/venv_mineru/bin/pip install 'mineru[core]'"
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    if MINERU_API_URL:
        progress_cb(0.02, f"Converting {input_path.name} via remote MinerU at {MINERU_API_URL}…")
    else:
        progress_cb(
            0.02,
            f"Converting {input_path.name} with MinerU "
            "(first run downloads model weights — can take a few minutes)…",
        )

    import queue
    import threading

    cmd = [str(MINERU_BIN), "-p", str(input_path), "-o", str(work_dir), "-b", MINERU_BACKEND, "-m", "auto"]
    if MINERU_API_URL:
        cmd += ["--api-url", MINERU_API_URL]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    output_lines: list[str] = []
    line_q: "queue.Queue[str | None]" = queue.Queue()

    def _pump() -> None:
        # MinerU's progress bars use carriage returns, not newlines, so read
        # raw characters and split on \r or \n to get one event per update.
        buf = ""
        assert proc.stdout is not None
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch in "\r\n":
                if buf.strip():
                    line_q.put(buf)
                buf = ""
            else:
                buf += ch
        if buf.strip():
            line_q.put(buf)
        line_q.put(None)  # sentinel: stream closed

    threading.Thread(target=_pump, daemon=True).start()

    elapsed = 0
    stream_done = False
    # Crawl from 0.05 to 0.35 over the conversion so the bar keeps visibly
    # moving even though MinerU doesn't report an exact percentage.
    while not stream_done or proc.poll() is None:
        frac = min(0.35, 0.05 + elapsed * 0.01)
        try:
            line = line_q.get(timeout=1.0)
        except queue.Empty:
            elapsed += 1
            progress_cb(frac, f"MinerU converting… ({elapsed}s elapsed)")
            continue
        if line is None:
            stream_done = True
            continue
        output_lines.append(line)
        progress_cb(frac, f"MinerU: {line.strip()[-120:]}")

    proc.wait()
    if proc.returncode != 0:
        tail = "\n".join(output_lines[-80:]).strip()
        raise IngestError(f"MinerU conversion failed (exit {proc.returncode}):\n{tail}")

    progress_cb(0.38, "MinerU conversion finished — locating output…")

    result = _locate_mineru_output(work_dir, input_path)
    if result is None:
        raise IngestError("MinerU finished but produced no markdown output.")
    return result


def _locate_mineru_output(work_dir: Path, input_path: Path) -> ConversionResult | None:
    """Find an existing MinerU conversion in work_dir, or None if absent.

    Used both to serve the cache (before running MinerU) and to locate output
    afterwards. Returns None when no markdown is present yet.
    """
    if not work_dir.exists():
        return None
    stem = input_path.stem
    candidates = sorted(work_dir.glob(f"{stem}/*/{stem}.md"))
    if not candidates:
        candidates = sorted(work_dir.rglob("*.md"))
    if not candidates:
        return None
    md_path = candidates[0]

    middle_candidates = sorted(md_path.parent.glob(f"{stem}_middle.json"))
    middle_json_path = middle_candidates[0] if middle_candidates else None

    # MinerU's "*_origin.pdf" is a byte-identical copy of the source PDF used
    # for OCR/layout — only produced when the input was itself a PDF.
    origin_candidates = sorted(md_path.parent.glob(f"{stem}_origin.pdf"))
    source_pdf_path = origin_candidates[0] if origin_candidates else (
        input_path if input_path.suffix.lower() == ".pdf" else None
    )

    return ConversionResult(md_path, middle_json_path, source_pdf_path)


def ingest_document(
    input_path: Path,
    llm_config,
    output_dir: Path,
    progress_cb: ProgressCB = _noop,
):
    """Full ingest: convert -> parse -> classify -> tables -> graph.

    Returns (graph, clause_map, clause_index, source_info):
      clause_map    {clause_id: Clause}                     — for the ReAct agent
      clause_index  {clause_id: {title, text_preview, category}} — for chat tools
      source_info   {"pdf_path": Path|None, "clause_pages": {clause_id: [{page, bbox}]}}
                     — for tracing/highlighting answers back to the source PDF
    """
    _ensure_pipeline_on_path()

    from regulatory_testgen.classify import classify_clauses
    from regulatory_testgen.graph import RegulatoryGraph
    from regulatory_testgen.extraction.tables import extract_tables
    from regulatory_testgen.parsing.markdown_parser import (
        parse_markdown_clauses,
        recover_missing_numeric_parents,
        save_clauses,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: convert — 0.00-0.40
    conversion = convert_to_markdown(
        input_path, output_dir / "mineru_out",
        progress_cb=lambda f, m: progress_cb(f * 0.40, m),
    )
    md_path = conversion.md_path

    # Stage 2: parse — 0.40-0.45
    progress_cb(0.40, "Parsing document structure…")
    from regulatory_testgen.structure_ids import namespace_clauses
    if conversion.source_pdf_path is not None and os.environ.get("REGULATORY_PARSER", "decoder") == "decoder":
        # clause hierarchy from the PDF layout, clause text (and tables) from MinerU
        from regulatory_testgen.parsing.pdf_parser import parse_pdf_clauses
        clauses = parse_pdf_clauses(conversion.source_pdf_path, md_path)
    else:
        clauses = parse_markdown_clauses(md_path)
        clauses = recover_missing_numeric_parents(clauses)
    clauses = namespace_clauses(clauses)  # annex clauses → namespaced ids (graph/index consistent)
    save_clauses(clauses, output_dir / "01_clauses.json")
    progress_cb(0.45, f"Parsed {len(clauses)} clauses.")

    # Stage 3: classify — 0.45-0.85 (cached: the LLM pass is the slowest stage,
    # so reuse a previous run's classifications for the same document).
    import json as _json

    classifications_path = output_dir / "01b_classifications.json"
    real_count = sum(1 for c in clauses if not c.is_pseudo_clause)

    if classifications_path.exists():
        progress_cb(0.85, "Using cached clause classifications.")
        classifications = _json.loads(classifications_path.read_text(encoding="utf-8"))
    else:
        def _classify_progress(batch_idx: int, total: int) -> None:
            frac = 0.45 + 0.40 * (batch_idx / total)
            progress_cb(frac, f"Classifying clauses with the LLM… batch {batch_idx}/{total}")

        progress_cb(0.45, f"Classifying {real_count} clauses with the LLM…")
        classifications = classify_clauses(clauses, llm_config, progress_cb=_classify_progress)
        classifications_path.write_text(
            _json.dumps(classifications, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Stage 4: tables — 0.85-0.92
    progress_cb(0.85, "Extracting performance tables…")
    tables = extract_tables(clauses)

    # Stage 5: graph — 0.92-1.00
    progress_cb(0.92, "Building knowledge graph…")
    graph = RegulatoryGraph.build(clauses, tables)

    clause_map: dict[str, Any] = {}
    for c in clauses:
        cid = c.clause_id
        if cid not in clause_map or len(c.text or "") > len(clause_map[cid].text or ""):
            clause_map[cid] = c

    clause_index = _build_clause_index(clauses, classifications)

    clause_pages: dict[str, list[dict]] = {}
    if conversion.middle_json_path is not None and conversion.source_pdf_path is not None:
        from pdf_highlight import build_clause_page_map
        clause_pages = build_clause_page_map(clauses, conversion.middle_json_path)
    source_info = {
        "pdf_path": conversion.source_pdf_path,
        "clause_pages": clause_pages,
        "middle_json_path": conversion.middle_json_path,
    }

    progress_cb(1.0, f"Ready — {graph.stats()['clauses']} clauses, {len(tables)} tables.")
    return graph, clause_map, clause_index, source_info


def _build_clause_index(clauses, classifications: dict[str, str]) -> dict[str, dict]:
    """Build the repaired clause index for an uploaded document.

    Converts the live Clause objects to the raw-dict shape and delegates to
    retrieval.build_clause_index, so uploads get the SAME treatment as the
    built-in doc: annex namespacing, duplicate/form-field cleanup, title
    recovery, and explicit `parent` links (needed by the structure tools,
    the folder tree, and the graph explorer)."""
    from retrieval import build_clause_index

    raw = [
        {
            "clause_id": c.clause_id,
            "uid": getattr(c, "uid", "") or "",  # classifications are keyed by uid
            "title": getattr(c, "title", "") or "",
            "text": getattr(c, "text", "") or "",
            "document_region": getattr(c, "document_region", "") or "",
            "section_path": list(getattr(c, "section_path", []) or []),
            "references": list(getattr(c, "references", []) or []),
            "is_pseudo_clause": getattr(c, "is_pseudo_clause", False),
            "line_start": getattr(c, "line_start", 0) or 0,
        }
        for c in clauses
    ]
    return build_clause_index(raw, classifications)
