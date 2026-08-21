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
    assert len(design["id"]) == 96 * 8
    assert np.all(np.bincount(design["id"]) == 8)
    assert np.all(np.bincount(design["group"]) == 96 * 2)
    assert np.linalg.matrix_rank(design["x"]) == design["x"].shape[1]


def test_small_condition_runs():
    result = MODULE.run_condition(
        n_subjects=96,
        effects=MODULE.SCENARIOS[0],
        replications=4,
        seed=20260820,
        inner_draws=100,
    )
    assert result["failures"] == 0
    assert 0.0 <= result["power_language_alpha025"] <= 1.0
    assert 0.0 <= result["power_action_alpha025"] <= 1.0
