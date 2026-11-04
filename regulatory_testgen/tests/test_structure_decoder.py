"""Tests for the PDF structure decoder and the PDF-structure clause parser.

Run from the thesis-code root, ideally in the pipeline venv:

    python -m unittest regulatory_testgen.tests.test_structure_decoder -v

The synthetic cases mirror failure modes met on real documents: a footnote marker
that is a legal successor (R152 p.11), a numbered table whose row numbers form a
longer chain than the real clauses (GTR 22), paragraphs deleted by amendments
(R131 12 -> 12.2), "Scenario N" headings standing in for clause N (R152 Annex 3
Appendix 2), and Roman/letter parts with continuous paragraph numbering (GTR 22
Part I). The end-to-end test runs on the bundled R152 PDF + MinerU Markdown.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from regulatory_testgen.parsing.structure_decoder import Line, decode_structure, successor_gap

BODY, H1 = 10.0, 14.0
DATA = Path(__file__).resolve().parents[2] / "data" / "R152r2E"


def body(page: int, text: str, size: float = BODY, bold: bool = False) -> Line:
    return Line(page=page, text=text, size=size, bold=bold, top=100.0, x0=85.0)


def ids(lines: list[Line], **kw) -> list[str]:
    return [h.clause_id for h in decode_structure(lines, **kw) if h.kind != "virtual"]


class SuccessorGapTest(unittest.TestCase):
    def test_legal_successors(self):
        self.assertEqual(successor_gap((5, 2, 1), (5, 2, 2)), 0)
        self.assertEqual(successor_gap((5, 2, 1), (5, 2, 1, 1)), 0)
        self.assertEqual(successor_gap((5, 2, 1, 4), (5, 3)), 0)
        self.assertEqual(successor_gap((5, 2, 1, 4), (6,)), 0)

    def test_gaps_and_missing_levels(self):
        self.assertEqual(successor_gap((12,), (12, 2)), 1)  # 12.1 deleted
        self.assertEqual(successor_gap((3,), (3, 1, 1)), 1)  # no 3.1 heading
        self.assertEqual(successor_gap((), (1, 1)), 1)  # appendix starting at 1.1
        self.assertIsNone(successor_gap((5, 2), (7, 1)))
        self.assertIsNone(successor_gap((6, 3, 3), (16,)))


class DecoderTest(unittest.TestCase):
    def test_footnote_successor_does_not_derail_the_chain(self):
        lines = [body(1, "1. Scope", H1, True), body(1, "1.1. Applies to M1."),
                 body(2, "2 For relative speeds between the listed values", 7.0),
                 body(2, "1.2. Second clause text."), body(3, "2. Definitions", H1, True),
                 body(3, "2.1. A definition.")]
        self.assertEqual(ids(lines), ["1", "1.1", "1.2", "2", "2.1"])

    def test_numbered_table_rows_lose_to_worded_clauses(self):
        # GTR 22 p.20: after 1.3.3 the table rows "2 ... 11" are a legal chain that is
        # longer than the real clauses 1.3.4 ... 2.1
        lines = [body(1, "1. In-use verification", H1, True)]
        lines += [body(1, f"{n}. Clause {n}") for n in ("1.1", "1.2", "1.3", "1.3.1", "1.3.2")]
        lines += [body(1, "1.3.3. Statistical method")]
        lines += [body(1, f"{n} 0.{n:03d} 0.438") for n in range(2, 12)]
        lines += [body(2, "1.3.4. Corrective measures"), body(2, "1.4. Part B verification"),
                  body(2, "1.4.1. Frequency of verifications"), body(2, "2. Rounding", H1, True),
                  body(2, "2.1. When the digit is less than 5")]
        got = ids(lines)
        self.assertIn("1.3.4", got)
        self.assertIn("2.1", got)
        self.assertNotIn("11", got)

    def test_deleted_paragraph_gap(self):
        # R131: 12.1 was deleted by an amendment, so 12 is followed by 12.2
        lines = [body(1, "1. Transitional provisions", H1, True),
                 body(1, "1.2. Transitional provisions applicable to the 02 series"),
                 body(1, "1.2.1. As from the official date of entry into force")]
        self.assertEqual(ids(lines), ["1", "1.2", "1.2.1"])

    def test_annex_container_and_implicit_scenario_parent(self):
        lines = [body(1, "1. Scope", H1, True), body(1, "1.1. Text."),
                 body(2, "Annex 3 - Appendix 2", 16.0, True), body(2, "False reaction scenarios", 16.0, True),
                 body(2, "Scenario 1", 12.0, True), body(2, "1.1. In this scenario the vehicle turns."),
                 body(2, "1.2. An example of the detail scenario:"),
                 body(3, "Scenario 2", 12.0, True), body(3, "2.1. In this scenario the vehicle follows."),
                 body(3, "2.2. An example of the detail scenario:")]
        heads = decode_structure(lines)
        got = {h.clause_id: h for h in heads}
        self.assertIn("Annex 3 - Appendix 2", got)
        self.assertIn("Annex 3 - Appendix 2 / 2.2", got)
        self.assertEqual(got["Annex 3 - Appendix 2 / 2"].kind, "virtual")
        self.assertEqual(got["Annex 3 - Appendix 2 / 2"].text, "Scenario 2")

    def test_roman_letter_parts_with_continuous_paragraphs(self):
        lines = [body(1, "I. Statement of technical rationale", H1, True), body(1, "A. Introduction", 12.0, True),
                 body(1, "1. Owing to the pressing need to reduce emissions"), body(1, "2. This GTR aims"),
                 body(2, "B. Procedural background", 12.0, True), body(2, "3. The informal working group"),
                 body(3, "II. Text of the GTR", H1, True), body(3, "1. Purpose", 12.0, True),
                 body(3, "1.1. This regulation provides"), body(3, "2. Scope", 12.0, True)]
        profile: dict = {}
        got = ids(lines, start=r"^I\.", profile=profile)
        self.assertEqual(set(got), {"I", "II", "I / A", "I / B", "I / 1", "I / 2", "I / 3",
                                    "II / 1", "II / 1.1", "II / 2"})
        self.assertIn((None, "roman"), profile["scopes"])

    def test_table_of_contents_is_skipped(self):
        lines = [body(1, "1. Scope ........................ 5"), body(1, "2. Definitions .................. 5"),
                 body(2, "1. Scope", H1, True), body(2, "1.1. Text."), body(2, "2. Definitions", H1, True)]
        self.assertEqual(ids(lines), ["1", "1.1", "2"])


@unittest.skipUnless(importlib.util.find_spec("fitz") and (DATA / "R152r2E_origin.pdf").exists(),
                     "needs PyMuPDF and the bundled R152 data")
class R152EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from regulatory_testgen.parsing.pdf_parser import parse_pdf_clauses

        cls.profile: dict = {}
        cls.clauses = parse_pdf_clauses(DATA / "R152r2E_origin.pdf", DATA / "R152r2E.md", profile=cls.profile)
        cls.by_id = {c.clause_id: c for c in cls.clauses}

    def test_clauses_the_markdown_parser_misses(self):
        for cid in ("5.1.3", "5.5.1", "7.2", "9.1", "Annex 3 - Appendix 2 / 2.1"):
            self.assertIn(cid, self.by_id)

    def test_hierarchy(self):
        self.assertEqual(self.by_id["5.1.4.1.1"].section_path[-1].split(" ")[0], "5.1.4.1")
        self.assertTrue(self.by_id["Annex 3 / 2.1"].section_path[0].startswith("Annex 3"))

    def test_tables_stay_with_their_clause(self):
        self.assertIn("<table", self.by_id["5.2.1.4"].text)

    def test_nearly_every_heading_aligns_to_the_markdown(self):
        self.assertLessEqual(len(self.profile["unaligned"]), 0.05 * len(self.clauses))
