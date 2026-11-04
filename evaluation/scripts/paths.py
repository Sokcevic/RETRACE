"""Locations used by the evaluation scripts; override with environment variables.

RETRACE_WORK  working directory with raw/ (benchmark downloads), runs/ (outputs, caches)
              default: evaluation/work
ADNQA_DATA    ADN-QA data folder (qa.jsonl, retrieval.jsonl, splits.json)
              default: datasets/adn-qa/data
The repository root is put on sys.path so that regulatory_testgen (the decoder) imports.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
W = Path(os.environ.get("RETRACE_WORK", REPO / "evaluation/work"))
ADNQA = Path(os.environ.get("ADNQA_DATA", REPO / "datasets/adn-qa/data"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
