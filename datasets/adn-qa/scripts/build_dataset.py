"""Assemble the ADN-QA dataset from the UNECE catalogue-of-questions files.

Layout written to <out>/:
  source_files/            every downloaded catalogue file, as published (docx + doc)
  source_files/SOURCES.tsv original URL, archive URL, sha256 per file
  data/questions.jsonl     2025 edition, one record per question, en/fr/de/ru aligned by number
  data/qa.jsonl            text-answerable multiple-choice questions (English)
  data/retrieval.jsonl     questions whose Source cites ADN paragraphs -> gold paragraph ids
  data/history.jsonl       English questions of the 2011-2023 editions (parsed, best effort)
  data/splits.json         dev / test question ids
  corpus/adn2025_structure.tsv  ADN 2025 paragraph ids, titles and parents (no body text)
  corpus/build_corpus.py   rebuilds the full-text corpus from the official ADN 2025 PDF

Usage: python -I build_dataset.py <catalogue download dir> <converted .doc dir> <paragraphs.jsonl> <out dir>
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_catalogue import parse_catalogue  # noqa: E402

LANGS = {"en": ("e", "e_0"), "de": ("de", "de"), "fr": ("f", "f"), "ru": ("r", "r")}
CATALOGUES = ("General", "Chemicals", "Gas")
EDITION_FILES = {  # English catalogue files per earlier edition, relative to the download dir
    2017: "docx/2017/dgwp15ac2/Catalogue_of_questions_2017/{c}/ADN_catalogue_of_questions_2017_{c}_E.docx",
    2019: "docx/2019/dgwp15ac2/Catalogue_of_questions_ADN_2019/{c}/ADN_catalogue_of_questions_2019_{c}{sep}e.docx",
    2021: "docx/2021-09/ADN_catalogue_of_questions_2021_{c}-e.docx",
    2023: "docx/2023-0{m}/ADN_catalogue_of_questions_2023_{c}_en.docx",
}
CONVERTED = {  # .doc editions converted to .docx with LibreOffice
    2011: {"General": "general2011e.docx", "Chemicals": "chemicals2011e.docx", "Gas": "gas2011e.docx"},
    2013: {"General": "general-2013-corr1e.docx", "Chemicals": "chemicals-2013e.docx", "Gas": "gas-2013e.docx"},
    2015: {"General": "General_2015_e.docx", "Chemicals": "Chemicals_2015_e.docx", "Gas": "Gas_2015_e.docx"},
}
DEV_FRACTION = 0.15
SEED = 13


def file_2025(src: Path, catalogue: str, lang: str) -> Path:
    month = "03" if catalogue == "General" else "02"
    suffix = LANGS[lang][1] if catalogue == "General" else LANGS[lang][0]
    return src / f"docx/2025-{month}/ADN_catalogue_of_questions_2025_{catalogue}_{suffix}.docx"


def mcq_ok(r: dict) -> bool:
    return not r["deleted"] and len(r["options"]) == 4 and r["answer"] in ("A", "B", "C", "D")


def build_questions(src: Path) -> list[dict]:
    per_lang: dict[str, dict[str, dict]] = {}
    for lang in LANGS:
        recs = {}
        for cat in CATALOGUES:
            for r in parse_catalogue(file_2025(src, cat, lang)):
                r["catalogue"] = cat
                recs[r["id"]] = r
        per_lang[lang] = recs
    out = []
    for qid, en in per_lang["en"].items():
        rec = {
            "id": qid,
            "edition": 2025,
            "catalogue": en["catalogue"],
            "section": en["catalogue_section"],
            "objective": en["objective"],
            "source": en["source"],
            "adn_refs": en["adn_refs"],
            "adn_refs_detailed": en["adn_refs_detailed"],
            "answer": en["answer"],
            "deleted": en["deleted"],
            "has_image": en["has_image"],
            "text": {},
        }
        for lang, recs in per_lang.items():
            r = recs.get(qid)
            if r is None:
                continue
            rec["text"][lang] = {
                "question": r["question"], "options": r["options"], "source": r["source"],
                "objective": r["objective"], "parse_ok": mcq_ok(r) and r["answer"] == en["answer"],
            }
        out.append(rec)
    return out


def build_history(src: Path, converted: Path) -> list[dict]:
    out = []
    for edition, pattern in EDITION_FILES.items():
        for cat in CATALOGUES:
            path = src / pattern.format(c=cat, sep="_-" if cat == "General" else "-",
                                        m="9" if cat == "General" else "2")
            out += [{**r, "edition": edition, "catalogue": cat, "file": path.name} for r in parse_catalogue(path)]
    for edition, files in CONVERTED.items():
        for cat, name in files.items():
            out += [{**r, "edition": edition, "catalogue": cat, "file": name}
                    for r in parse_catalogue(converted / name)]
    return out


def copy_sources(src: Path, dst: Path, url_lists: list[Path]) -> None:
    urls = {}
    for ul in url_lists:
        for u in ul.read_text().split():
            urls[re.sub(r"^.*/", "", u.replace("%C3%A8", "è"))] = u
    rows = ["path\turl\tarchive_url\tsha256"]
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = f.relative_to(src)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / rel)
        url = urls.get(f.name, "")
        rows.append(f"{rel}\t{url}\thttps://web.archive.org/web/2026id_/{url}\t"
                    f"{hashlib.sha256(f.read_bytes()).hexdigest()}")
    (dst / "SOURCES.tsv").write_text("\n".join(rows) + "\n")


def main() -> None:
    src, converted, paragraphs, out = map(Path, sys.argv[1:5])
    url_lists = [Path(p) for p in sys.argv[5:]]
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "corpus").mkdir(parents=True, exist_ok=True)

    questions = build_questions(src)
    corpus = [json.loads(l) for l in paragraphs.open()]
    ids = {p["id"] for p in corpus}

    qa, retrieval = [], []
    for q in questions:
        en = q["text"]["en"]
        if mcq_ok({"deleted": q["deleted"], "options": en["options"], "answer": q["answer"]}) and not q["has_image"]:
            qa.append({"id": q["id"], "catalogue": q["catalogue"], "question": en["question"],
                       "options": en["options"], "answer": q["answer"],
                       "answer_text": en["options"][q["answer"]], "adn_refs": q["adn_refs"]})
        if not q["deleted"] and q["adn_refs"]:
            retrieval.append({"id": q["id"], "catalogue": q["catalogue"], "query": en["question"],
                              "gold": q["adn_refs"], "gold_detailed": q["adn_refs_detailed"],
                              "source": q["source"],
                              "gold_in_structure": [g in ids for g in q["adn_refs"]]})

    rng = random.Random(SEED)
    live = sorted(q["id"] for q in questions if not q["deleted"])
    dev = set(rng.sample(live, round(DEV_FRACTION * len(live))))
    splits = {"dev": sorted(dev), "test": [i for i in live if i not in dev]}

    def dump(name, rows):
        with (out / "data" / name).open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("questions.jsonl", questions)
    dump("qa.jsonl", qa)
    dump("retrieval.jsonl", retrieval)
    dump("history.jsonl", build_history(src, converted))
    (out / "data" / "splits.json").write_text(json.dumps(splits, indent=1))

    with (out / "corpus" / "adn2025_structure.tsv").open("w") as f:
        f.write("id\tparent\ttitle\n")
        for p in corpus:
            f.write(f"{p['id']}\t{p['parent'] or ''}\t{p['title'].replace(chr(9), ' ')}\n")

    copy_sources(src, out / "source_files", url_lists)

    stats = {
        "questions_total": len(questions),
        "questions_live": len(live),
        "qa_text_mcq": len(qa),
        "retrieval_queries": len(retrieval),
        "retrieval_distinct_gold": len({g for r in retrieval for g in r["gold"]}),
        "retrieval_all_gold_in_structure": sum(all(r["gold_in_structure"]) for r in retrieval),
        "per_catalogue": Counter(q["catalogue"] for q in questions if not q["deleted"]),
        "lang_parse_ok": {l: sum(1 for q in questions if q["text"].get(l, {}).get("parse_ok")) for l in LANGS},
        "answer_distribution": Counter(r["answer"] for r in qa),
        "dev": len(splits["dev"]), "test": len(splits["test"]),
    }
    (out / "data" / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
