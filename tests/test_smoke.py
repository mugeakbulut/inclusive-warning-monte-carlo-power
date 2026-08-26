from pathlib import Path
import importlib.util
import sys

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "wp3_mediation_power.py"
SPEC = importlib.util.spec_from_file_location("wp3_mediation_power", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_balanced_design():
    design = MODULE.build_design(96)
    assert design["fixed"].shape[:2] == (96, 8)
    assert design["z"].shape == (96, 8, 4)
    assert np.allclose(design["z"][0, :, 1:].mean(axis=0), 0)


def test_small_condition_runs():
    result = MODULE.run_condition(
        n_subjects=96,
        effects=MODULE.SCENARIOS[0],
        replications=4,
        seed=MODULE.BASE_SEED,
        inner_draws=100,
    )
    assert result["failures"] == 0
    assert 0.0 <= result["power_language_alpha025"] <= 1.0
    assert 0.0 <= result["power_action_alpha025"] <= 1.0
    assert result["random_slope_sd"] == MODULE.PRIMARY_RANDOM_SLOPE_SD
