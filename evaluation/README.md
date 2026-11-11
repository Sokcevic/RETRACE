# RETRACE evaluation

Scripts behind the REFSQ 2027 paper: retrieval and QA on ADN-QA, ObliQA, LegalBench-RAG and QASPER.

## Setup

    python -m venv .venv && .venv/bin/pip install -r evaluation/requirements.txt -r regulatory_testgen/requirements.txt
    export RETRACE_WORK=$PWD/evaluation/work   # raw/ (benchmark downloads) and runs/ (outputs, caches)
    export ADNQA_DATA=$PWD/datasets/adn-qa/data

The benchmarks are not in the repository. Put them under `$RETRACE_WORK/raw/` (obliqa, lbrag, qasper);
`eval_retrieval.py` documents the expected layout. ADN-QA source files: `datasets/adn-qa/scripts/download_sources.py`.
No API keys are stored here. The reader LLM is any OpenAI-compatible server (paper: Qwen3-32B, FP8, on vLLM):

    docker run --runtime nvidia -p 127.0.0.1:8010:8000 -v /models/Qwen3-32B:/model/Qwen3-32B:ro \
      vllm/vllm-openai --model /model/Qwen3-32B --served-model-name qwen3-32b --quantization fp8 --max-model-len 16384
    export RETRACE_LLM_URL=http://localhost:8010/v1

### PDF parser

`RETRACE_PARSER=pymupdf` (default) feeds the structure decoder with PyMuPDF lines.
`RETRACE_PARSER=mineru` uses MinerU's reading order (fixes two-column papers) with fonts from the PDF
(`regulatory_testgen/parsing/mineru_lines.py`). Run MinerU first:

    mineru -p <pdf dir> -o $RETRACE_WORK/raw/mineru/<dataset> -b pipeline -m auto   # MinerU 3.4.0, pypdfium2 4.30.0

Outputs of a non-default parser get a suffix, e.g. `runs/methods_qasper_mineru.json`.

## Scripts (run from `evaluation/scripts`)

| Script | Output (`runs/`) | What |
|---|---|---|
| `analyze_structure.py` | `structure_analysis.json`, `xref_<ds>.json` | structure statistics per corpus, cross-reference in-degree of gold sections |
| `eval_methods.py <ds> [--small]` | `methods_<ds>.json/.pkl` | matched word budgets: vanilla (flat-N), RETRACE (struct-N, paragraph), no-header ablation |
| `xref_join.py` | `xref_join.json` | hit@1000w by how often the gold section is cited |
| `eval_links.py <adn\|obliqa>` | `links_<ds>.json` | ablation: number of cross-reference links followed |
| `eval_paragraph.py` | `paragraph_adn_<backend>.json` | ADN paragraph-level retrieval and QA |
| `eval_retrieval.py`, `rescore_adn.py`, `significance.py` | | chunk-level retrieval, corrected ADN gold, paired bootstrap |
| `eval_qa_methods.py <adn\|qasper>` | `qa_methods_<ds>_<model>.json` | answer quality at matched budgets: closed-book, vanilla, DOS RAG, RETRACE, oracle |
| `eval_qa.py` | `qa_*.json` | earlier reader QA (top-5 chunks); holds the shared `LLM` client |

The outputs behind the paper's tables are in `evaluation/results/` (see its README).

`rag_core.py` holds chunking (flat windows, structure chunks from the decoder), BM25, dense, RRF and the hit rule.
