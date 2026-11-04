"""Download the 190 UNECE ADN question catalogues listed in source_files/SOURCES.tsv
and check their sha256. unece.org blocks automated clients, so the Wayback Machine
copy (archive_url) is used when the original URL fails.

Usage: python download_sources.py [--dest ../source_files]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
UA = "Mozilla/5.0 (compatible; research dataset rebuild)"


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
            return r.read()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=HERE / "source_files")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(HERE / "source_files/SOURCES.tsv"), delimiter="\t"))
    bad = 0
    for r in rows:
        out = args.dest / r["path"]
        if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == r["sha256"]:
            continue
        for url in (r["url"], r["archive_url"]):
            data = fetch(url)
            if data and hashlib.sha256(data).hexdigest() == r["sha256"]:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
                break
            time.sleep(1)
        else:
            bad += 1
            print("failed:", r["path"])
    print(f"{len(rows) - bad}/{len(rows)} files present and verified")


if __name__ == "__main__":
    main()
