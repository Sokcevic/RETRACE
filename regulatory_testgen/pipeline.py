"""Five-stage pipeline for UN regulatory test case generation.

Stage 1: Parse    — PDF structure + MinerU Markdown text → List[Clause]
Stage 1b: Classify — LLM semantic clause classification
Stage 2: Tables   — Clause text → List[RegulationTable]
Stage 3: Graph    — LlamaIndex PropertyGraphStore
Stage 4: Generate — Dynamiq ReAct agent × testable clauses

Each stage writes a JSON checkpoint to output_dir so the pipeline can be
resumed or inspected after any stage.

The parallel generation in Stage 4 uses a ThreadPoolExecutor capped at
config.workers. Each worker runs an independent agent instance to avoid
shared state issues.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from regulatory_testgen.agent import analyse_clause, build_agent
from regulatory_testgen.classify import (
    TESTABLE_CATEGORIES,
    category_of,
    classify_clauses,
    is_testable,
    load_classifications,
    save_classifications,
)
from regulatory_testgen.config import PipelineConfig
from regulatory_testgen.export import save_csv, save_json, save_markdown
from regulatory_testgen.graph import RegulatoryGraph
from regulatory_testgen.models import TestCase
from regulatory_testgen.extraction.tables import extract_tables
from regulatory_testgen.data_models.core import Clause
from regulatory_testgen.parsing.markdown_parser import (
    load_clauses,
    parse_markdown_clauses,
    recover_missing_numeric_parents,
    save_clauses,
)
from regulatory_testgen.parsing.pdf_parser import parse_pdf_clauses

# Titles that MinerU strips from section headings; used by recover_missing_numeric_parents.
_SECTION_TITLE_HINTS: dict[str, str] = {
    "5.2":  "Specific Requirements",
    "6.1":  "Test Conditions",
    "6.3":  "Test Targets",
    "6.7":  "Warning and Activation Test with a Bicycle Target",
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(markdown_path: str | Path, config: PipelineConfig | None = None) -> list[TestCase]:
    """Run the full pipeline and return the generated test cases.

    Intermediate results are written to config.output_dir:
      01_clauses.json       — parsed clause objects
      02_graph_summary.json — graph node/edge statistics
      03_test_cases.json    — full TestCase objects
      03_test_cases.csv     — 3-column CSV matching ground truth format
    """
    effective = config or PipelineConfig()
    out = Path(effective.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Stage 1 — Parse -------------------------------------------------------
    logger.info("Stage 1: Parsing %s", markdown_path)
    clauses = parse_clauses(markdown_path, effective)
    save_clauses(clauses, out / "01_clauses.json")
    logger.info("  → %d clauses extracted", len(clauses))

    # Stage 1b — Classify ---------------------------------------------------
    classifications_path = out / "01b_classifications.json"
    if not effective.use_llm_classification:
        # Ablation mode: mark every non-pseudo clause as "obligation" so all
        # are passed to the generator, bypassing the semantic filter entirely.
        logger.info("Stage 1b: SKIPPED (use_llm_classification=False) — all non-pseudo clauses treated as testable")
        classifications = {
            c.clause_id: ("formatting" if c.is_pseudo_clause else "obligation")
            for c in clauses
        }
    elif classifications_path.exists():
        logger.info("Stage 1b: Loading cached clause classifications")
        classifications = load_classifications(classifications_path)
    else:
        logger.info("Stage 1b: Classifying clauses with LLM")
        classifications = classify_clauses(clauses, effective.llm)
        save_classifications(classifications, classifications_path)
    category_counts = {}
    for v in classifications.values():
        cat = category_of(v)
        category_counts[cat] = category_counts.get(cat, 0) + 1
    logger.info("  → %s", ", ".join(f"{k}:{v}" for k, v in sorted(category_counts.items())))

    # Stage 2 — Extract tables ----------------------------------------------
    logger.info("Stage 2: Extracting tables")
    tables = extract_tables(clauses)
    logger.info("  → %d performance tables extracted", len(tables))

    # Stage 3 — Build knowledge graph ----------------------------------------
    logger.info("Stage 3: Building knowledge graph")
    graph = RegulatoryGraph.build(clauses, tables)
    graph.save(out / "02_graph_summary.json")
    stats = graph.stats()
    logger.info(
        "  → %d clause nodes, %d REFERS_TO edges, %d CONTAINS edges",
        stats["clauses"],
        stats["refers_to_edges"],
        stats["contains_edges"],
    )

    # Stage 4 — Generate test cases ------------------------------------------
    logger.info("Stage 4: Generating test cases")
    testable = _select_testable_clauses(clauses, classifications)
    logger.info("  → %d/%d clauses selected for generation", len(testable), len(clauses))

    test_cases = _generate_parallel(testable, graph, config)
    logger.info("  → %d test cases generated", len(test_cases))

    # Export -----------------------------------------------------------------
    save_json(test_cases, out / "03_test_cases.json")       # includes markdown field
    save_csv(test_cases, out / "03_test_cases.csv")
    save_markdown(test_cases, out / "03_test_cases.md")
    logger.info("Results written to %s", out)

    return test_cases


def parse_clauses(input_path: str | Path, config: PipelineConfig) -> list[Clause]:
    """Stage 1. input_path is a MinerU Markdown file or the source PDF itself.

    With parser="decoder" the clause hierarchy comes from the PDF and the text
    from the Markdown (PDF text if there is no Markdown); otherwise, or when no
    source PDF can be found, the original MinerU Markdown parser runs."""
    path = Path(input_path)
    markdown = path if path.suffix.lower() in {".md", ".markdown"} else None
    if config.pdf_path:
        pdf = Path(config.pdf_path)
    elif path.suffix.lower() == ".pdf":
        pdf = path
    else:
        pdf = find_source_pdf(path)

    if config.parser == "decoder":
        if pdf is not None and pdf.exists():
            logger.info("  structure from %s%s", pdf, f", text from {markdown}" if markdown else "")
            return parse_pdf_clauses(pdf, markdown)
        logger.warning("No source PDF found for %s; falling back to the Markdown parser", path)
    if markdown is None:
        raise ValueError(f"The Markdown parser needs a MinerU Markdown file, got {path}")
    clauses = parse_markdown_clauses(markdown)
    return recover_missing_numeric_parents(clauses, title_hints=_SECTION_TITLE_HINTS)


def find_source_pdf(markdown_path: Path) -> Path | None:
    """MinerU writes the source PDF next to its Markdown as <stem>_origin.pdf."""
    for candidate in (markdown_path.with_name(f"{markdown_path.stem}_origin.pdf"),
                      markdown_path.with_suffix(".pdf")):
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_testable_clauses(
    clauses: list[Clause],
    classifications: dict[str, str],
) -> list[Clause]:
    """Filter clauses to those whose LLM-assigned category is testable.

    Testable categories: obligation, test_procedure, performance_data.
    Clauses without a classification entry (shouldn't happen) are excluded.
    The ReAct agent makes the final per-clause decision during generation.
    """
    return [
        c
        for c in clauses
        if not c.is_pseudo_clause
        and is_testable(classifications.get(getattr(c, "uid", None)) or classifications.get(c.clause_id))
        and len((c.text or "").strip()) > 20
    ]


def _generate_parallel(
    clauses: list[Clause],
    graph: RegulatoryGraph,
    config: PipelineConfig,
) -> list[TestCase]:
    """Generate test cases for all candidate clauses using a worker pool.

    Each worker gets its own Agent instance to avoid shared state in the
    Dynamiq prompt history.
    """
    all_cases: list[TestCase] = []
    total = len(clauses)

    def process(index: int, clause: Clause) -> tuple[int, list[TestCase]]:
        agent = build_agent(graph, config.llm)
        cases = analyse_clause(agent, clause.clause_id, clause.title, clause.text)
        # Fallback: if the ReAct loop timed out without producing output, retry
        # once with a direct prompt that skips tool calls (no context retrieval).
        if not cases:
            logger.info("  Retrying %s with direct prompt (no tool calls)", clause.clause_id)
            agent2 = build_agent(graph, config.llm)
            cases = analyse_clause(
                agent2,
                clause.clause_id,
                clause.title,
                clause.text,
                direct=True,
            )
        return index, cases

    worker_count = min(config.workers, total)
    completed = 0

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(process, i, c): c.clause_id
            for i, c in enumerate(clauses)
        }
        for future in as_completed(futures):
            clause_id = futures[future]
            try:
                _, cases = future.result()
                all_cases.extend(cases)
                completed += 1
                logger.info(
                    "  [%d/%d] %s → %d test case(s)",
                    completed,
                    total,
                    clause_id,
                    len(cases),
                )
            except Exception as exc:
                completed += 1
                logger.warning("  [%d/%d] %s failed: %s", completed, total, clause_id, exc)

    return all_cases


# ---------------------------------------------------------------------------
# Resume helpers (load intermediate results without re-running earlier stages)
# ---------------------------------------------------------------------------


def load_clauses_checkpoint(output_dir: str | Path) -> list[Clause]:
    return load_clauses(Path(output_dir) / "01_clauses.json")


def load_test_cases_checkpoint(output_dir: str | Path) -> list[TestCase]:
    data = json.loads((Path(output_dir) / "03_test_cases.json").read_text(encoding="utf-8"))
    return [TestCase.model_validate(item) for item in data]
