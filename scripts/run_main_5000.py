#!/usr/bin/env python3
"""N=288 ana aracılık koşullarını üç ilişkisiz eğimle 5.000 kez çalıştırır."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wp3_mediation_power import (  # noqa: E402
    BASE_SEED,
    PRIMARY_RANDOM_SLOPE_SD,
    SCENARIOS,
    run_condition,
)


def run_one(item):
    index, scenario = item
    return run_condition(
        288, scenario, 5000, BASE_SEED + index * 101,
        icc=0.50, slope_sd=PRIMARY_RANDOM_SLOPE_SD, inner_draws=3000,
    )


def main():
    with ProcessPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(run_one, enumerate(SCENARIOS)))
    out = ROOT / "results" / "ana_guc_sonuclari.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
