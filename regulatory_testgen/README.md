# regulatory_testgen

Agentic LLM system for generating UN-compliant test cases from regulatory PDFs, using
Dynamiq (agent orchestration) and LlamaIndex (Graph-RAG).

## Architecture

```
MinerU PDF → Markdown            source PDF
      ↓                               ↓
      ↓            structure decoder (parsing/structure_decoder.py)
      ↓                               ↓  clause ids + hierarchy
Stage 1: Parse (parsing/pdf_parser.py: text from Markdown, structure from PDF)
      ↓  Clause objects
Stage 2: Tables (extraction/tables.py)
      ↓  14 RegulationTable objects
Stage 3: Build Knowledge Graph (LlamaIndex PropertyGraphStore)
      ↓  196 CLAUSE nodes, 83 REFERS_TO edges, 155 CONTAINS edges
Stage 4: Generate test cases (Dynamiq ReAct agent × 84 testable clauses)
      ↓
03_test_cases.json  +  03_test_cases.csv
```

### Key design decisions

| Decision | Rationale |
|---|---|
| LlamaIndex PropertyGraphStore | Typed REFERS_TO / CONTAINS edge labels; built-in `get_triplets` for BFS |
| BFS transitive closure | Guarantees complete cross-reference context (BifrostRAG +12.3 F1) |
| Dynamiq ReAct Agent | Interleaved reasoning + tool calls; agent can follow unexpected refs |
| Template + instantiation | Table-driven clauses: one LLM template × N rows = N identical-structure tests |
| Clause structure from the PDF | MinerU flattens every heading to one level and drops/splits clause numbers; the decoder reads the PDF layout and keeps the globally most consistent chain of clause numbers instead |

## Setup

```bash
# Python 3.12 required (Dynamiq caps at <3.14)
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# Full pipeline
venv/bin/python -m regulatory_testgen run \
    /path/to/R152r2E.md \
    --output-dir output

# Point at a different OpenAI-compatible endpoint (hosted or self-hosted vLLM)
venv/bin/python -m regulatory_testgen run \
    /path/to/R152r2E.md \
    --base-url http://your-vllm-host:8000/v1 \
    --model your-model-id

# Inspect graph stats after running
venv/bin/python -m regulatory_testgen graph-stats --output-dir output

# Reproduce the thesis' MinerU-only parsing (the bundled checkpoints in output/ use it)
venv/bin/python -m regulatory_testgen run /path/to/R152r2E.md --parser markdown
```

### Stage 1 parsers

By default (`--parser decoder`) the clause hierarchy comes from the source PDF:
`parsing/structure_decoder.py` reads the layout with PyMuPDF, treats every line
that starts with a number (`5.2.1`, `A1.2`, `II.`, `B.`, `(2)`) as a candidate,
and keeps the best chain of legal successors over the whole document, so footnote
markers, numbered table rows and figure labels cannot break the tree. Annexes and
appendices namespace their clauses (`Annex 3 / 5.1`) like `structure_ids.py`.
Clause text, and with it the HTML tables Stage 2 reads, still comes from MinerU's
Markdown: each decoded heading is aligned to its Markdown line.

The PDF is found as `--pdf`, as the input itself (PDF-only: text from the PDF), or
as MinerU's `<stem>_origin.pdf` next to the Markdown. Without one the pipeline
falls back to the Markdown parser. If the PDF is tagged and `pypdf` is installed,
text inside tagged tables is ignored. In the chat UI, uploads use the decoder when
MinerU produced a source PDF; set `REGULATORY_PARSER=markdown` to switch it off.

On R152 the decoder finds 268 clauses (the Markdown parser: 231), every clause's
parent is present in the graph, and Stage 2 extracts the same 13 tables.

## Output

| File | Contents |
|---|---|
| `01_clauses.json` | Parsed clause objects (269 for R152 with the decoder, 253 with `--parser markdown`) |
| `01b_classifications.json` | LLM clause category classifications |
| `02_graph_summary.json` | Graph statistics + clause index |
| `03_test_cases.json` | Full TestCase objects with traceability |
| `03_test_cases.csv` | 3-column CSV: `Unique_ID, Test_title, test_scenario` |

`output/` ships pre-populated with the R152 checkpoints so `chat_ui/` works without
re-running the pipeline.

## Package structure

```
regulatory_testgen/
├── __init__.py           version info
├── config.py             LLMConfig, PipelineConfig
├── models.py             TestCase, ClauseRole
├── graph.py              RegulatoryGraph — LlamaIndex PropertyGraphStore + BFS
├── prompts.py            LLM prompt templates + few-shot examples
├── agent.py              Dynamiq tools (GetClause, GetReferencedClauses, GetTables) + agent factory
├── classify.py           LLM-based clause classification
├── pipeline.py           4-stage orchestration with JSON checkpoints
├── export.py             JSON + CSV writers
├── cli.py                argparse CLI (`python -m regulatory_testgen`)
├── parsing/              PDF structure decoder + Markdown → Clause objects
│                         (structure_decoder.py, pdf_parser.py, markdown_parser.py, tree_builder.py)
├── extraction/           Clause text → tables/references/requirements
├── data_models/          Clause, DocumentTree, RegulationTable, and related dataclasses
└── output/               Pre-computed checkpoints for the bundled R152 regulation
```
