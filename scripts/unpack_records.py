#!/usr/bin/env python3
"""Decompress the released per-run records (results/<run>/**/rows.json.gz -> rows.json) so that the analysis scripts can read them.

    python3 scripts/unpack_records.py results/campaign_2026-10-09_1120
    python3 scripts/merge_results.py results/campaign_2026-10-09_1120          # every table, figure and test of the run, re-derived
    python3 plotting/make_plots.py --run results/campaign_2026-10-09_1120 --out plotting/assets_rebuilt   # the paper's tables and figures

The .gz files are kept; rows.json is only written when it is missing or older than the archive."""
import gzip, os, shutil, sys


def main(run):
    n = 0
    for d, _, files in os.walk(run):
        for f in files:
            if f == "rows.json.gz":
                src = os.path.join(d, f); dst = src[:-3]
                if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src): continue
                with gzip.open(src, "rb") as fin, open(dst, "wb") as fout: shutil.copyfileobj(fin, fout, 1 << 24)
                n += 1
    print(f"unpacked {n} record file(s) under {run}")


if __name__ == "__main__":
    if len(sys.argv) != 2: sys.exit(__doc__)
    main(sys.argv[1])
