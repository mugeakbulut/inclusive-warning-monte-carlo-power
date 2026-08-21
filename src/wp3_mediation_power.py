#!/usr/bin/env python3
"""WP3 2x2x2 tekrarlı ölçümlü aracılık için Monte Carlo güç simülasyonu.

Bu kod yalnızca açık kaynak Python bağımlılıklarını kullanır. Dengeli eksik
blok atamasını, kişi içi merkezlemeyi ve katılımcı kümeli sağlam standart
hataları içerir. Dolaylı etkiler normal-çarpım Monte Carlo güven aralıklarıyla
değerlendirilir.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class EffectScenario:
    name: str
    a_language: float
    a_action: float
    b_within: float


SCENARIOS = (
    EffectScenario("Temkinli", 0.10, 0.10, 0.20),
    EffectScenario("Orta", 0.20, 0.20, 0.30),
    EffectScenario("İyimser", 0.30, 0.30, 0.40),
)


def balanced_latin_order(subject: int) -> np.ndarray:
    """Sekiz hücre için dengeli, terslemeli Latin-kare sırası."""
    base = np.array([0, 1, 7, 2, 6, 3, 5, 4], dtype=int)
    if subject % 2:
        base = base[::-1]
    return (base + (subject // 2)) % 8


def build_design(n_subjects: int) -> dict[str, np.ndarray]:
    if n_subjects % 4:
        raise ValueError("Örneklem büyüklüğü dört gruba bölünebilmelidir.")

    cells = np.array(
        [(l, a, f) for l in (-0.5, 0.5) for a in (-0.5, 0.5) for f in (-0.5, 0.5)],
        dtype=float,
    )
    ids, groups, language, action, focus, scenario, order = [], [], [], [], [], [], []

    for sid in range(n_subjects):
        seq = balanced_latin_order(sid)
        order_for_cell = np.empty(8, dtype=int)
        order_for_cell[seq] = np.arange(1, 9)
        # Her kişi her temel senaryoyu iki kez görür. Her hücre, çalışma
        # genelinde dört senaryonun her biriyle dengeli biçimde eşleşir.
        scenario_for_cell = (np.arange(8) + sid) % 4
        for cell in range(8):
            ids.append(sid)
            groups.append(sid % 4)
            language.append(cells[cell, 0])
            action.append(cells[cell, 1])
            focus.append(cells[cell, 2])
            scenario.append(scenario_for_cell[cell])
            order.append(order_for_cell[cell])

    sid = np.asarray(ids, dtype=int)
    language = np.asarray(language)
    action = np.asarray(action)
    focus = np.asarray(focus)
    scenario = np.asarray(scenario, dtype=int)
    order = np.asarray(order, dtype=float)
    order_z = (order - 4.5) / np.std(np.arange(1, 9), ddof=0)

    scenario_dummies = np.column_stack([(scenario == k).astype(float) for k in (1, 2, 3)])
    x = np.column_stack(
        [
            language,
            action,
            focus,
            language * action,
            language * focus,
            action * focus,
            language * action * focus,
            scenario_dummies,
            order_z,
        ]
    )
    x = within_center(x, n_subjects)
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise RuntimeError("Tasarım matrisi tam sıralı değil; atama planını kontrol edin.")

    return {
        "id": sid,
        "group": np.asarray(groups, dtype=int),
        "language": language,
        "action": action,
        "focus": focus,
        "scenario": scenario,
        "order_z": order_z,
        "x": x,
    }


def within_center(values: np.ndarray, n_subjects: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    one_dim = arr.ndim == 1
    if one_dim:
        arr = arr[:, None]
    shaped = arr.reshape(n_subjects, 8, arr.shape[1])
    centered = shaped - shaped.mean(axis=1, keepdims=True)
    out = centered.reshape(arr.shape)
    return out[:, 0] if one_dim else out


def ols_cluster(x: np.ndarray, y: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS ve katılımcı düzeyinde CR1 sağlam standart hata."""
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    n, k = x.shape
    groups = int(ids.max()) + 1
    scores = np.zeros((groups, k))
    np.add.at(scores, ids, x * resid[:, None])
    meat = scores.T @ scores
    correction = (groups / (groups - 1)) * ((n - 1) / (n - k))
    cov = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return beta, se


def generate_data(
    design: dict[str, np.ndarray],
    effects: EffectScenario,
    rng: np.random.Generator,
    icc: float = 0.50,
    random_slope_sd: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    n_subjects = int(design["id"].max()) + 1
    ids = design["id"]
    l = design["language"]
    a = design["action"]
    f = design["focus"]
    s = design["scenario"]
    order_z = design["order_z"]

    # Standartlaştırılmış bileşik puanlar. ICC toplam varyansın katılımcı
    # düzeyindeki payını belirler. Dört senaryo sabit etki olarak üretilir.
    u = rng.multivariate_normal(
        mean=[0.0, 0.0],
        cov=[[icc, 0.30 * icc], [0.30 * icc, icc]],
        size=n_subjects,
    )
    rs_l = rng.normal(0.0, random_slope_sd, n_subjects)
    rs_a = rng.normal(0.0, random_slope_sd, n_subjects)
    residual_sd = math.sqrt(max(1.0 - icc, 0.05))
    scenario_m = np.array([-0.12, -0.04, 0.04, 0.12])
    scenario_y = np.array([-0.10, -0.03, 0.03, 0.10])

    mediator = (
        effects.a_language * l
        + effects.a_action * a
        + 0.15 * f
        + scenario_m[s]
        - 0.02 * order_z
        + u[ids, 0]
        + rs_l[ids] * l
        + rs_a[ids] * a
        + rng.normal(0.0, residual_sd, len(ids))
    )
    mediator_wc = within_center(mediator, n_subjects)

    outcome = (
        0.10 * l
        + 0.10 * a
        + 0.10 * f
        + 0.08 * (l * a)
        + effects.b_within * mediator_wc
        + scenario_y[s]
        - 0.02 * order_z
        + u[ids, 1]
        + rng.normal(0.0, residual_sd, len(ids))
    )
    return mediator, outcome


def fit_paths(
    design: dict[str, np.ndarray], mediator: np.ndarray, outcome: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_subjects = int(design["id"].max()) + 1
    ids = design["id"]
    x = design["x"]
    mediator_wc = within_center(mediator, n_subjects)
    outcome_wc = within_center(outcome, n_subjects)
    beta_m, se_m = ols_cluster(x, mediator_wc, ids)
    x_y = np.column_stack([x, mediator_wc])
    beta_y, se_y = ols_cluster(x_y, outcome_wc, ids)
    # Dil a-yolu, eylem a-yolu ve kişi-içi b-yolu.
    estimates = np.array([beta_m[0], beta_m[1], beta_y[-1]])
    ses = np.array([se_m[0], se_m[1], se_y[-1]])
    return estimates, ses


def product_ci_detect(
    estimates: np.ndarray,
    ses: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    inner_draws: int,
) -> tuple[bool, bool]:
    """İki a×b etkisi için Monte Carlo ürün güven aralığı."""
    b_draw = rng.normal(estimates[2], ses[2], inner_draws)
    detected = []
    for idx in (0, 1):
        a_draw = rng.normal(estimates[idx], ses[idx], inner_draws)
        product = a_draw * b_draw
        lo, hi = np.quantile(product, [alpha / 2, 1 - alpha / 2])
        detected.append(bool(lo > 0 or hi < 0))
    return detected[0], detected[1]


def run_condition(
    n_subjects: int,
    effects: EffectScenario,
    replications: int,
    seed: int,
    icc: float = 0.50,
    random_slope_sd: float = 0.0,
    inner_draws: int = 3000,
) -> dict[str, float | int | str]:
    design = build_design(n_subjects)
    rng = np.random.default_rng(seed)
    detections_05 = np.zeros((replications, 2), dtype=bool)
    detections_025 = np.zeros((replications, 2), dtype=bool)
    joint_sig_05 = np.zeros((replications, 2), dtype=bool)
    estimates_all = np.zeros((replications, 3))
    failures = 0

    for r in range(replications):
        try:
            mediator, outcome = generate_data(
                design, effects, rng, icc=icc, random_slope_sd=random_slope_sd
            )
            estimates, ses = fit_paths(design, mediator, outcome)
            if not np.all(np.isfinite(estimates)) or not np.all(np.isfinite(ses)):
                raise FloatingPointError("Sonlu olmayan katsayı veya standart hata")
            estimates_all[r] = estimates
            detections_05[r] = product_ci_detect(estimates, ses, 0.05, rng, inner_draws)
            detections_025[r] = product_ci_detect(estimates, ses, 0.025, rng, inner_draws)
            pvals = 2 * norm.sf(np.abs(estimates / ses))
            joint_sig_05[r] = [(pvals[0] < 0.05 and pvals[2] < 0.05),
                               (pvals[1] < 0.05 and pvals[2] < 0.05)]
        except Exception:
            failures += 1

    valid = replications - failures
    if valid <= 0:
        raise RuntimeError("Bütün simülasyon tekrarları başarısız oldu.")

    # Başarısız tekrarlar güç hesabında tespit yok olarak tutulur.
    p_l_05, p_a_05 = detections_05.mean(axis=0)
    p_l_025, p_a_025 = detections_025.mean(axis=0)
    both_05 = np.logical_and(detections_05[:, 0], detections_05[:, 1]).mean()
    both_025 = np.logical_and(detections_025[:, 0], detections_025[:, 1]).mean()
    mcse_l = math.sqrt(max(p_l_025 * (1 - p_l_025) / replications, 0.0))
    mcse_a = math.sqrt(max(p_a_025 * (1 - p_a_025) / replications, 0.0))

    return {
        "scenario": effects.name,
        "n_complete": n_subjects,
        "n_per_group": n_subjects // 4,
        "n_recruit_15pct_loss": int(math.ceil((n_subjects / 0.85) / 4) * 4),
        "a_language": effects.a_language,
        "a_action": effects.a_action,
        "b_within": effects.b_within,
        "indirect_language": effects.a_language * effects.b_within,
        "indirect_action": effects.a_action * effects.b_within,
        "icc": icc,
        "random_slope_sd": random_slope_sd,
        "replications": replications,
        "inner_draws": inner_draws,
        "power_language_alpha05": p_l_05,
        "power_action_alpha05": p_a_05,
        "power_both_alpha05": both_05,
        "power_language_alpha025": p_l_025,
        "power_action_alpha025": p_a_025,
        "power_both_alpha025": both_025,
        "mcse_language_alpha025": mcse_l,
        "mcse_action_alpha025": mcse_a,
        "joint_sig_language_alpha05": joint_sig_05[:, 0].mean(),
        "joint_sig_action_alpha05": joint_sig_05[:, 1].mean(),
        "mean_a_language_hat": estimates_all[:, 0].mean(),
        "mean_a_action_hat": estimates_all[:, 1].mean(),
        "mean_b_hat": estimates_all[:, 2].mean(),
        "failures": failures,
        "seed": seed,
    }


def run_full(output_dir: Path, quick: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_seed = 20260820
    main_reps = 300 if quick else 5000
    grid_reps = 150 if quick else 800
    sens_reps = 200 if quick else 1000
    null_reps = 200 if quick else 1000
    inner = 800 if quick else 3000

    main_rows = []
    for i, scenario in enumerate(SCENARIOS):
        main_rows.append(
            run_condition(288, scenario, main_reps, base_seed + i * 101, inner_draws=inner)
        )
    pd.DataFrame(main_rows).to_csv(output_dir / "ana_guc_sonuclari.csv", index=False)

    grid_ns = list(range(96, 481, 16))
    grid_rows = []
    for s_idx, scenario in enumerate(SCENARIOS):
        for n_idx, n in enumerate(grid_ns):
            grid_rows.append(
                run_condition(
                    n,
                    scenario,
                    grid_reps,
                    base_seed + 10000 + s_idx * 1000 + n_idx * 17,
                    inner_draws=inner,
                )
            )
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(output_dir / "orneklem_tarama_sonuclari.csv", index=False)

    sensitivity_rows = []
    medium = SCENARIOS[1]
    for j, icc in enumerate((0.30, 0.50, 0.70)):
        for k, slope_sd in enumerate((0.0, 0.15)):
            sensitivity_rows.append(
                run_condition(
                    288,
                    medium,
                    sens_reps,
                    base_seed + 20000 + j * 100 + k * 13,
                    icc=icc,
                    random_slope_sd=slope_sd,
                    inner_draws=inner,
                )
            )
    pd.DataFrame(sensitivity_rows).to_csv(output_dir / "duyarlilik_sonuclari.csv", index=False)

    null_rows = []
    null_cases = (
        EffectScenario("Dil a-yolu sıfır", 0.0, 0.20, 0.30),
        EffectScenario("Eylem a-yolu sıfır", 0.20, 0.0, 0.30),
        EffectScenario("b-yolu sıfır", 0.20, 0.20, 0.0),
    )
    for i, case in enumerate(null_cases):
        null_rows.append(
            run_condition(
                288,
                case,
                null_reps,
                base_seed + 30000 + i * 101,
                inner_draws=inner,
            )
        )
    pd.DataFrame(null_rows).to_csv(output_dir / "yanlis_pozitif_sonuclari.csv", index=False)

    thresholds = []
    for scenario in SCENARIOS:
        sub = grid_df[grid_df["scenario"] == scenario.name].sort_values("n_complete")
        eligible = sub[
            (sub["power_language_alpha025"] >= 0.80)
            & (sub["power_action_alpha025"] >= 0.80)
        ]
        if len(eligible):
            row = eligible.iloc[0]
            thresholds.append(
                {
                    "scenario": scenario.name,
                    "minimum_n_grid": int(row["n_complete"]),
                    "minimum_per_group": int(row["n_per_group"]),
                    "recruit_with_15pct_loss": int(row["n_recruit_15pct_loss"]),
                }
            )
        else:
            thresholds.append(
                {
                    "scenario": scenario.name,
                    "minimum_n_grid": None,
                    "minimum_per_group": None,
                    "recruit_with_15pct_loss": None,
                }
            )
    pd.DataFrame(thresholds).to_csv(output_dir / "orneklem_esikleri.csv", index=False)

    metadata = {
        "analysis_date": "2026-08-20",
        "base_seed": base_seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "design": "2x2x2 within-person; 4 groups; 4 fixed scenarios; 8 observations/person",
        "primary_alpha": 0.025,
        "replication_counts": {
            "main_n288": main_reps,
            "sample_size_grid": grid_reps,
            "sensitivity": sens_reps,
            "null_checks": null_reps,
            "product_draws_per_replication": inner,
        },
        "notes": [
            "Etki büyüklükleri pilot veri değil, standartlaştırılmış planlama varsayımlarıdır.",
            "Güvenilirlik ve davranışsal niyet madde ortalamaları sürekli bileşik puan olarak temsil edilmiştir.",
            "Kişi içi etkiler katılımcı sabit etkisiyle giderilmiş ve katılımcı kümeli CR1 standart hatalar kullanılmıştır.",
            "Ürün güven aralığında a ve b tahminlerinin normal örnekleme dağılımları ve sıfır tahmin kovaryansı varsayılmıştır.",
        ],
        "thresholds": thresholds,
    }
    (output_dir / "simulasyon_bilgileri.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="mc_aracilik_sonuclari")
    parser.add_argument("--quick", action="store_true", help="Yalnızca kod kontrolü için kısa koşu")
    args = parser.parse_args()
    run_full(Path(args.output_dir), quick=args.quick)


if __name__ == "__main__":
    main()
