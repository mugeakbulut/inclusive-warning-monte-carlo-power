#!/usr/bin/env python3
"""WP3 2x2x2 çok düzeyli aracılık için Monte Carlo güç simülasyonu.

Birincil model her katılımcı için rastgele kesişim ile sadelik, eylem
yönlendirmesi ve odak değişkenlerinin birbiriyle ilişkisiz rastgele
eğimlerini içerir. H5a ve H5b, a1*b ve a2*b ürünlerinin Monte Carlo güven
aralıklarıyla sınanır.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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

ROWS_PER_SUBJECT = 8
PRIMARY_RANDOM_SLOPE_SD = 0.15
BASE_SEED = 20260826


def balanced_latin_order(subject: int) -> np.ndarray:
    base = np.array([0, 1, 7, 2, 6, 3, 5, 4], dtype=int)
    if subject % 2:
        base = base[::-1]
    return (base + subject // 2) % 8


def build_design(n_subjects: int) -> dict[str, np.ndarray]:
    """Dört eşit grup ve kişi başına sekiz 2x2x2 gözlem oluşturur."""
    if n_subjects % 4:
        raise ValueError("Örneklem büyüklüğü dört gruba tam bölünmelidir.")

    cells = np.array(
        [(l, a, f) for l in (-0.5, 0.5) for a in (-0.5, 0.5) for f in (-0.5, 0.5)],
        dtype=float,
    )
    l = np.tile(cells[:, 0], (n_subjects, 1))
    a = np.tile(cells[:, 1], (n_subjects, 1))
    f = np.tile(cells[:, 2], (n_subjects, 1))
    scenario = np.empty((n_subjects, 8), dtype=int)
    order = np.empty((n_subjects, 8), dtype=float)
    for sid in range(n_subjects):
        scenario[sid] = (np.arange(8) + sid) % 4
        seq = balanced_latin_order(sid)
        order[sid, seq] = np.arange(1, 9)
    order_z = (order - 4.5) / np.std(np.arange(1, 9), ddof=0)

    scenario_dummies = np.stack([(scenario == k).astype(float) for k in (1, 2, 3)], axis=2)
    group = np.arange(n_subjects) % 4
    group_dummies = np.stack([(group == k).astype(float) for k in (1, 2, 3)], axis=1)
    group_dummies = np.repeat(group_dummies[:, None, :], 8, axis=1)
    fixed = np.concatenate(
        [
            np.ones((n_subjects, 8, 1)),
            l[:, :, None], a[:, :, None], f[:, :, None],
            (l * a)[:, :, None], (l * f)[:, :, None], (a * f)[:, :, None],
            (l * a * f)[:, :, None], scenario_dummies, order_z[:, :, None], group_dummies,
        ],
        axis=2,
    )
    z = np.stack([np.ones_like(l), l, a, f], axis=2)
    return {"language": l, "action": a, "focus": f, "scenario": scenario,
            "order_z": order_z, "fixed": fixed, "z": z}


def working_precision(z: np.ndarray, icc: float, slope_sd: float) -> np.ndarray:
    """Diyagonal G ile V=(ZGZ'+R) çalışma kovaryansının tersini verir."""
    residual_var = max(1.0 - icc, 0.05)
    g = np.diag([icc, slope_sd**2, slope_sd**2, slope_sd**2])
    v = z @ g @ z.T + residual_var * np.eye(8)
    return np.linalg.inv(v)


def generate_data(design, effects, rng, icc=0.50, slope_sd=PRIMARY_RANDOM_SLOPE_SD):
    n = design["language"].shape[0]
    l, a, f, s, oz = (design[k] for k in ("language", "action", "focus", "scenario", "order_z"))
    # Aracı ve sonuç rastgele kesişimleri orta düzeyde ilişkilidir; üç eğim
    # her denklem içinde ve denklemler arasında bağımsızdır.
    intercepts = rng.multivariate_normal([0, 0], [[icc, 0.30 * icc], [0.30 * icc, icc]], n)
    slopes_m = rng.normal(0, slope_sd, (n, 3))
    slopes_y = rng.normal(0, slope_sd, (n, 3))
    residual_sd = math.sqrt(max(1.0 - icc, 0.05))
    scenario_m = np.array([-0.12, -0.04, 0.04, 0.12])
    scenario_y = np.array([-0.10, -0.03, 0.03, 0.10])

    m = (effects.a_language*l + effects.a_action*a + 0.15*f + scenario_m[s] - 0.02*oz
         + intercepts[:, 0, None] + slopes_m[:, 0, None]*l
         + slopes_m[:, 1, None]*a + slopes_m[:, 2, None]*f
         + rng.normal(0, residual_sd, (n, 8)))
    m_wc = m - m.mean(axis=1, keepdims=True)
    y = (0.10*l + 0.10*a + 0.10*f + 0.08*l*a + effects.b_within*m_wc
         + scenario_y[s] - 0.02*oz + intercepts[:, 1, None]
         + slopes_y[:, 0, None]*l + slopes_y[:, 1, None]*a + slopes_y[:, 2, None]*f
         + rng.normal(0, residual_sd, (n, 8)))
    return m, y


def gls_cluster(x: np.ndarray, y: np.ndarray, precision: np.ndarray):
    """Ortak kişi-içi V ile GLS ve katılımcı kümeli CR1 kovaryansı."""
    # x: kişi x 8 x p; y: kişi x 8
    bread_inv = np.linalg.pinv(np.einsum("nip,ij,njq->pq", x, precision, x))
    rhs = np.einsum("nip,ij,nj->p", x, precision, y)
    beta = bread_inv @ rhs
    resid = y - np.einsum("nip,p->ni", x, beta)
    scores = np.einsum("nip,ij,nj->np", x, precision, resid)
    n, p = scores.shape
    correction = n / (n - 1) * ((n * 8 - 1) / (n * 8 - p))
    cov = correction * bread_inv @ (scores.T @ scores) @ bread_inv
    influence = scores @ bread_inv.T
    return beta, cov, influence


def fit_paths(design, mediator, outcome, precision):
    n = mediator.shape[0]
    x_m = design["fixed"]
    beta_m, cov_m, infl_m = gls_cluster(x_m, mediator, precision)
    m_wc = mediator - mediator.mean(axis=1, keepdims=True)
    m_pm = np.repeat(mediator.mean(axis=1, keepdims=True), 8, axis=1)
    x_y = np.concatenate([design["fixed"], m_wc[:, :, None], m_pm[:, :, None]], axis=2)
    beta_y, cov_y, infl_y = gls_cluster(x_y, outcome, precision)
    # fixed sütunları: 0 kesişim, 1 sadelik, 2 eylem, 3 odak, ...
    idx_a = (1, 2)
    idx_b = x_y.shape[2] - 2
    estimates = np.array([beta_m[idx_a[0]], beta_m[idx_a[1]], beta_y[idx_b]])
    cov = np.zeros((3, 3))
    cov[:2, :2] = cov_m[np.ix_(idx_a, idx_a)]
    cov[2, 2] = cov_y[idx_b, idx_b]
    # Aynı katılımcılardan elde edilen iki denklemin yol tahminleri arasındaki
    # küme düzeyi örnekleme kovaryansını koru.
    correction = n / (n - 1)
    cross = correction * infl_m[:, idx_a].T @ infl_y[:, idx_b]
    cov[:2, 2] = cross
    cov[2, :2] = cross
    return estimates, cov


def product_ci_detect(estimates, covariance, alpha, rng, inner_draws):
    cov = (covariance + covariance.T) / 2
    eigval, eigvec = np.linalg.eigh(cov)
    cov = eigvec @ np.diag(np.maximum(eigval, 1e-12)) @ eigvec.T
    draws = rng.multivariate_normal(estimates, cov, inner_draws)
    products = draws[:, :2] * draws[:, 2, None]
    q = np.quantile(products, [alpha/2, 1-alpha/2], axis=0)
    return (q[0] > 0) | (q[1] < 0)


def run_condition(n_subjects, effects, replications, seed, icc=0.50,
                  slope_sd=PRIMARY_RANDOM_SLOPE_SD, inner_draws=3000):
    design = build_design(n_subjects)
    precision = working_precision(design["z"][0], icc, slope_sd)
    rng = np.random.default_rng(seed)
    detected_05 = np.zeros((replications, 2), bool)
    detected_025 = np.zeros((replications, 2), bool)
    estimates_sum = np.zeros(3)
    failures = 0
    for r in range(replications):
        try:
            m, y = generate_data(design, effects, rng, icc, slope_sd)
            est, cov = fit_paths(design, m, y, precision)
            if not np.all(np.isfinite(est)) or not np.all(np.isfinite(cov)):
                raise FloatingPointError
            estimates_sum += est
            detected_05[r] = product_ci_detect(est, cov, 0.05, rng, inner_draws)
            detected_025[r] = product_ci_detect(est, cov, 0.025, rng, inner_draws)
        except (np.linalg.LinAlgError, FloatingPointError):
            failures += 1
    valid = replications - failures
    mean_est = estimates_sum / max(valid, 1)
    p05 = detected_05.mean(axis=0)
    p025 = detected_025.mean(axis=0)
    return {
        "scenario": effects.name, "n_complete": n_subjects,
        "n_per_group": n_subjects // 4,
        "n_recruit_15pct_loss": int(math.ceil((n_subjects / 0.85) / 4) * 4),
        "a_language": effects.a_language, "a_action": effects.a_action,
        "b_within": effects.b_within,
        "indirect_language": effects.a_language * effects.b_within,
        "indirect_action": effects.a_action * effects.b_within,
        "icc": icc, "random_slope_sd": slope_sd,
        "random_effects": "intercept + language + action + focus; diagonal G",
        "replications": replications, "inner_draws": inner_draws,
        "power_language_alpha05": p05[0], "power_action_alpha05": p05[1],
        "power_both_alpha05": np.logical_and(detected_05[:, 0], detected_05[:, 1]).mean(),
        "power_language_alpha025": p025[0], "power_action_alpha025": p025[1],
        "power_both_alpha025": np.logical_and(detected_025[:, 0], detected_025[:, 1]).mean(),
        "mcse_language_alpha025": math.sqrt(p025[0]*(1-p025[0])/replications),
        "mcse_action_alpha025": math.sqrt(p025[1]*(1-p025[1])/replications),
        "mean_a_language_hat": mean_est[0], "mean_a_action_hat": mean_est[1],
        "mean_b_hat": mean_est[2], "failures": failures, "seed": seed,
    }


def _run_job(args):
    return run_condition(*args)


def run_jobs(jobs, workers):
    if workers == 1:
        return [_run_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_run_job, jobs))


def run_full(output_dir: Path, quick: bool = False, workers: int | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or min(8, os.cpu_count() or 1)
    main_reps, grid_reps = ((120, 50) if quick else (5000, 600))
    sens_reps, null_reps = ((60, 60) if quick else (1000, 1000))
    inner = 600 if quick else 3000

    main_jobs = [(288, s, main_reps, BASE_SEED+i*101, .50, PRIMARY_RANDOM_SLOPE_SD, inner)
                 for i, s in enumerate(SCENARIOS)]
    main = run_jobs(main_jobs, workers)
    pd.DataFrame(main).to_csv(output_dir / "ana_guc_sonuclari.csv", index=False)

    grid_ns = (192, 208, 224, 240, 248, 256, 272, 288, 304, 320, 336, 352)
    grid_jobs = [(n, s, grid_reps, BASE_SEED+10000+i*1000+j*17,
                  .50, PRIMARY_RANDOM_SLOPE_SD, inner)
                 for i, s in enumerate(SCENARIOS) for j, n in enumerate(grid_ns)]
    grid = run_jobs(grid_jobs, workers)
    grid_df = pd.DataFrame(grid)
    grid_df.to_csv(output_dir / "orneklem_tarama_sonuclari.csv", index=False)

    sensitivity_jobs = [(288, SCENARIOS[0], sens_reps, BASE_SEED+20000+j*31,
                         .50, sd, inner)
                        for j, sd in enumerate((0.10, 0.15, 0.20, 0.25))]
    sensitivity = run_jobs(sensitivity_jobs, workers)
    pd.DataFrame(sensitivity).to_csv(output_dir / "duyarlilik_sonuclari.csv", index=False)

    null_cases = (EffectScenario("Sadelik a-yolu sıfır", 0, .20, .30),
                  EffectScenario("Eylem a-yolu sıfır", .20, 0, .30),
                  EffectScenario("b-yolu sıfır", .20, .20, 0))
    null_jobs = [(288, s, null_reps, BASE_SEED+30000+i*101,
                  .50, PRIMARY_RANDOM_SLOPE_SD, inner)
                 for i, s in enumerate(null_cases)]
    null = run_jobs(null_jobs, workers)
    pd.DataFrame(null).to_csv(output_dir / "yanlis_pozitif_sonuclari.csv", index=False)

    thresholds = []
    for s in SCENARIOS:
        sub = grid_df[grid_df.scenario == s.name].sort_values("n_complete")
        ok = sub[(sub.power_language_alpha025 >= .80) & (sub.power_action_alpha025 >= .80)]
        thresholds.append({"scenario": s.name,
                           "minimum_n_grid": None if ok.empty else int(ok.iloc[0].n_complete),
                           "minimum_per_group": None if ok.empty else int(ok.iloc[0].n_per_group),
                           "recruit_with_15pct_loss": None if ok.empty else int(ok.iloc[0].n_recruit_15pct_loss)})
    pd.DataFrame(thresholds).to_csv(output_dir / "orneklem_esikleri.csv", index=False)

    metadata = {
        "analysis_date": "2026-08-26", "base_seed": BASE_SEED,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "design": "2x2x2 within-person; 4 equal groups; 4 fixed scenarios; 8 observations/person",
        "primary_model": "participant random intercept plus three uncorrelated random slopes (language, action, focus)",
        "primary_random_slope_sd": PRIMARY_RANDOM_SLOPE_SD,
        "primary_alpha_per_indirect_effect": 0.025,
        "simplification_rule": [
            "Remove the random slope with the smallest estimated variance.",
            "Repeat until the model is nonsingular and converges.",
            "For exact ties remove focus, then action, then language; always retain random intercept."
        ],
        "replication_counts": {"main_n288": main_reps, "sample_size_grid": grid_reps,
                               "sensitivity": sens_reps, "null_checks": null_reps,
                               "product_draws_per_replication": inner},
        "parallel_workers": workers,
        "notes": [
            "Effect sizes and variance components are planning assumptions, not pilot estimates.",
            "Reliability and behavioral-intention item means are represented as standardized continuous scores.",
            "Power is the proportion of replications whose Monte Carlo product confidence interval excludes zero.",
            "The joint sampling covariance of a1, a2 and b is retained using participant-cluster influence functions."
        ], "thresholds": thresholds,
    }
    (output_dir / "simulasyon_bilgileri.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run_full(Path(args.output_dir), args.quick, args.workers)


if __name__ == "__main__":
    main()
