"""Command-line interface for regulatory_testgen.

Usage examples:

  # Full pipeline (all stages)
  python -m regulatory_testgen run path/to/R152r2E.md

  # Custom output directory and model/endpoint
  python -m regulatory_testgen run R152r2E.md \\
      --output-dir output \\
      --model gpt-4.1-mini \\
      --base-url https://api.openai.com/v1

  # Increase parallelism for faster generation
  python -m regulatory_testgen run R152r2E.md --workers 8

  # Clause structure comes from the source PDF by default (R152r2E_origin.pdf
  # next to the Markdown); reproduce the thesis' MinerU-only parsing with:
  python -m regulatory_testgen run R152r2E.md --parser markdown

  # Inspect intermediate graph stats
  python -m regulatory_testgen graph-stats --output-dir output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> None:
    from regulatory_testgen.config import LLMConfig, PipelineConfig
    from regulatory_testgen.pipeline import run_pipeline

    llm_config = LLMConfig(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_loops=args.max_loops,
    )
    config = PipelineConfig(
        llm=llm_config,
        output_dir=args.output_dir,
        workers=args.workers,
        parser=args.parser,
        pdf_path=args.pdf,
    )

    test_cases = run_pipeline(args.markdown, config)
    print(f"\nDone. {len(test_cases)} test case(s) written to {args.output_dir}/")


def cmd_graph_stats(args: argparse.Namespace) -> None:
    summary_path = Path(args.output_dir) / "02_graph_summary.json"
    if not summary_path.exists():
        print(f"No graph summary found at {summary_path}. Run the pipeline first.")
        sys.exit(1)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = data.get("stats", {})
    print("\nGraph statistics:")
    for k, v in stats.items():
        print(f"  {k:30s} {v}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="regulatory_testgen",
        description="Agentic UN regulatory test case generation",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = sub.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument("markdown", help="Path to MinerU Markdown output file (or the source PDF)")
    run_parser.add_argument("--output-dir", default="output")
    run_parser.add_argument(
        "--parser",
        choices=["decoder", "markdown"],
        default="decoder",
        help="decoder: clause structure from the PDF (default); markdown: MinerU-only parser",
    )
    run_parser.add_argument("--pdf", default=None, help="Source PDF (default: <stem>_origin.pdf next to the Markdown)")
    run_parser.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
    )
    run_parser.add_argument("--model", default="gpt-4.1-mini")
    run_parser.add_argument("--temperature", type=float, default=0.1)
    run_parser.add_argument("--max-tokens", type=int, default=8192)
    run_parser.add_argument("--max-loops", type=int, default=10)
    run_parser.add_argument("--workers", type=int, default=512)
    run_parser.set_defaults(func=cmd_run)

    # --- graph-stats ---
    gs_parser = sub.add_parser("graph-stats", help="Print knowledge graph statistics")
    gs_parser.add_argument("--output-dir", default="output")
    gs_parser.set_defaults(func=cmd_graph_stats)

    parsed = parser.parse_args(argv)
    _setup_logging(parsed.verbose)
    parsed.func(parsed)


if __name__ == "__main__":
    main()
