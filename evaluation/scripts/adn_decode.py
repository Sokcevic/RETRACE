"""Decode the ADN 2025 PDF and report structure coverage of catalogue refs."""
import collections, json, pickle, sys, time
from pathlib import Path
import paths  # noqa: F401  (puts the repo root on sys.path)
from regulatory_testgen.parsing.structure_decoder import pdf_layout_lines, decode_structure

from paths import W  # noqa: E402
cache = W / "runs/adn_lines.pkl"
if cache.exists() and "--fresh" not in sys.argv:
    lines = pickle.load(open(cache, "rb"))
else:
    t = time.time(); lines = pdf_layout_lines(W / "raw/adn/adn2025_en.pdf"); print("layout", round(time.time()-t), "s")
    pickle.dump(lines, open(cache, "wb"))
prof = {}
hs = decode_structure(lines, profile=prof)
print("lines", len(lines), "headings", len(hs), "profile", {k: v for k, v in prof.items() if k != "scopes"})
ids = [h.clause_id for h in hs]
json.dump([dict(id=h.clause_id, label=h.label, kind=h.kind, page=lines[h.line_index].page, text=h.text[:80]) for h in hs],
          open(W / "runs/adn_headings.json", "w"), indent=0)
print(collections.Counter(i.split(" / ")[0] if " / " in i else i.split(".")[0] for i in ids).most_common(25))
