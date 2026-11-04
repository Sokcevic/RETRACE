# ADN-QA: retrieval and question answering over the ADN dangerous-goods regulations

ADN-QA turns the official UNECE/CCNR *Catalogue of questions* for the ADN expert
examination into a benchmark for retrieval-augmented question answering over a
long, deeply structured regulation. The ADN (European Agreement concerning the
International Carriage of Dangerous Goods by Inland Waterways) 2025 edition has
1,146 pages and its paragraphs are numbered up to six levels deep (e.g. 7.2.4.16.8).

The catalogue is written by regulators for the examination of ADN experts. Every
question is multiple choice with four options, one correct answer, and a
**Source** field that, for about half of the questions, cites the ADN paragraph(s) the
answer comes from. This gives two tasks with expert-written gold labels:

| Task | File | Size | Gold |
|---|---|---|---|
| Question answering (multiple choice) | `data/qa.jsonl` | 1,223 questions | correct option (A–D) |
| Retrieval | `data/retrieval.jsonl` | 646 queries, 354 distinct ADN paragraphs | cited ADN paragraph ids |

The 2025 edition covers three catalogues: General (691 live questions), Chemicals
(296) and Gas (265). Questions are aligned across the four official languages
(English, French, German, Russian) by their catalogue number.

## Files

> **Not included in this repository:** `data/` and the corpus files are withheld until
> UNECE confirms redistribution rights (see *Licence and permissions*). Rebuild them with
> `scripts/download_sources.py`, `scripts/parse_catalogue.py` and `scripts/build_dataset.py`.

```
data/questions.jsonl      2025 edition, all 1,325 numbered questions (incl. 73 deleted),
                          with en/fr/de/ru text, objective, Source, cited ADN refs, answer
data/qa.jsonl             English, text-only multiple-choice questions (deleted and
                          picture-based questions removed)
data/retrieval.jsonl      questions whose Source cites ADN paragraphs; gold = paragraph ids
data/splits.json          dev (188) / test (1,064) question ids, seed 13
data/history.jsonl        English questions of the 2011, 2013, 2015, 2017, 2019, 2021,
                          2023 editions (parsed best-effort; General 2011/2013 partial)
data/stats.json           counts
corpus/adn2025_structure.tsv   paragraph tree of ADN 2025 (id, parent, title), no body text
corpus/build_corpus.py         rebuilds the full-text paragraph corpus from the official PDF
source_files/             every catalogue file as published (190 .docx/.doc files, all
                          editions and languages, incl. summary and directive documents);
                          not in git: fetch them with scripts/download_sources.py
source_files/SOURCES.tsv  original URL, Wayback Machine URL and SHA-256 of each file
scripts/                  parser and dataset builder
```

### Record formats

`qa.jsonl`
```json
{"id": "110 01.0-02", "catalogue": "General", "question": "What does ADN govern?",
 "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "answer": "B",
 "answer_text": "...", "adn_refs": ["1.1.2.1"]}
```

`retrieval.jsonl`
```json
{"id": "110 01.0-03", "catalogue": "General", "query": "Where do the classes of dangerous goods appear in ADN?",
 "gold": ["2.1.1.1"], "gold_detailed": [{"paragraph": "2.1.1.1", "item": null}],
 "source": "2.1.1.1", "gold_in_structure": [true]}
```
`gold` holds ADN paragraph numbers; `gold_detailed` keeps sub-items such as
`8.1.2.1 (g)`. `gold_in_structure` says whether the paragraph is present in
`corpus/adn2025_structure.tsv` (true for 566 of 646 queries; the rest cite
Chapter 3.2 / Table C material, forms, or numbers that do not exist in ADN 2025,
such as `1.02`).

## The ADN text (corpus)

The ADN 2025 regulations are © United Nations and are not redistributed here.
`corpus/build_corpus.py` downloads nothing itself; give it the official PDF
(*ADN 2025 English.pdf*, UNECE, January 2025) and it rebuilds the paragraph
corpus (`id, title, text, parent, references`) with the RETRACE structure decoder.
`corpus/adn2025_structure.tsv` lists the resulting paragraph ids so that results
can be compared without the text.

## Construction

1. The catalogue page (unece.org/transport/road-transport/catalogue-questions)
   links 124 .docx and 66 .doc files. unece.org blocks scripted downloads, so all
   files were taken from the Wayback Machine (`SOURCES.tsv`).
2. `scripts/parse_catalogue.py` reads the Word tables (number | Source | answer, then
   the question with four options), handles options split across rows or typed in one
   paragraph, Cyrillic look-alike option letters in the Russian files, deleted
   questions and picture-only options. `.doc` editions were converted with LibreOffice.
3. ADN paragraph references are extracted from the Source field
   (e.g. `8.1.2.1 (g), 7.2.3.7.1.5` → `8.1.2.1`, `7.2.3.7.1.5`).

Known limitations: 27 questions contain pictures (11 have picture-only options and
are excluded from `qa.jsonl`); the French, German and Russian texts parse slightly
less cleanly than the English one (`parse_ok` per language in `questions.jsonl`);
the Source field cites paragraphs at different granularities (section vs. paragraph).

## Licence and permissions

The catalogue of questions is published by UNECE / the Central Commission for the
Navigation of the Rhine. **Redistribution rights for the source files and the
question texts have to be confirmed with the UNECE secretariat before public
release.** The parsing scripts and annotations are released under MIT.

## Citation

TBD (REFSQ 2027 submission).
