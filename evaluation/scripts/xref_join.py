"""Does it matter whether the gold section is cross-referenced elsewhere in its document?

Joins runs/xref_<ds>.json (in-degree of the gold section, analyze_structure.py)
with the per-query hits of runs/methods_<ds>.pkl (eval_methods.py) and reports
hit@1000w of vanilla (flat-200, dense) and RETRACE (paragraph, hybrid / dense)
for queries whose gold section is cited by 0, 1-2 or 3+ other sections, and
for gold in front matter (text before the first decoded heading).

Usage: python xref_join.py [ds ...] -> runs/xref_join.json
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

from paths import W  # noqa: E402
import rag_core  # noqa: E402
SYSTEMS = [("flat-200", "dense"), ("flat-200", "hybrid"), ("struct-200", "hybrid"), ("paragraph", "dense"),
           ("paragraph", "hybrid")]
METRIC = "hit@1000w"


def group(r: dict) -> str:
    if r["in_front_matter"] and not r["gold_sections"]:
        return "front matter"
    d = r["indegree"]
    return "cited 0x" if d == 0 else "cited 1-2x" if d <= 2 else "cited 3+x"


def main():
    out = {}
    for ds in sys.argv[1:] or ["adn", "obliqa", "lbrag", "qasper"]:
        t = rag_core.tag()
        xp, mp = W / f"runs/xref_{ds}{t}.json", W / f"runs/methods_{ds}{t}.pkl"
        if not (xp.exists() and mp.exists()):
            print("skip", ds)
            continue
        xr = {r["id"]: r for r in json.load(open(xp))}
        m = pickle.load(open(mp, "rb"))
        groups = [group(xr[i]) for i in m["ids"]]
        out[ds] = {}
        for g in ("cited 0x", "cited 1-2x", "cited 3+x", "front matter"):
            idx = [k for k, x in enumerate(groups) if x == g]
            if not idx:
                continue
            out[ds][g] = {"n": len(idx)} | {f"{s} [{r}]": round(float(m["per"][(s, r)][METRIC][idx].mean()), 3)
                                             for s, r in SYSTEMS if (s, r) in m["per"]}
        print(ds)
        for g, v in out[ds].items():
            print(f"  {g:13s}", v)
    (W / f"runs/xref_join{rag_core.tag()}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
