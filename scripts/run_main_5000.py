#!/usr/bin/env python3
"""N=288 ana aracılık koşullarını 5.000 tekrarla yeniden çalıştırır."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wp3_mediation_power import SCENARIOS, run_condition  # noqa: E402


def run_one(item):
    index, scenario = item
    return run_condition(
        n_subjects=288,
        effects=scenario,
        replications=5000,
        seed=20260820 + index * 101,
        inner_draws=3000,
    )


def main():
    with ProcessPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(run_one, enumerate(SCENARIOS)))

    results_dir = ROOT / "results"
    pd.DataFrame(rows).to_csv(results_dir / "ana_guc_sonuclari.csv", index=False)

    metadata_path = results_dir / "simulasyon_bilgileri.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["replication_counts"] = {
        "main_n288": 5000,
        "sample_size_grid": 800,
        "sensitivity": 1000,
        "null_checks": 1000,
        "product_draws_per_replication": 3000,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
