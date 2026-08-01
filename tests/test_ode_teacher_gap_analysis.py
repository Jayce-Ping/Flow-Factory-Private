import numpy as np
import pytest

from scripts.xopd_analysis.analyze_ode_teacher_gap_images import (
    COMMON_EPOCHS,
    RUN_LABELS,
    _paired_bootstrap_epoch80,
)
from scripts.xopd_analysis.analyze_ode_teacher_gap_probes import _linear_cka, _rbf_mmd
from scripts.xopd_analysis.probe_ode_teacher_gap import _parse_ints


def test_parse_ints_fails_fast_on_duplicates_and_bad_values():
    with pytest.raises(ValueError, match="unique integers"):
        _parse_ints("0,4,4", "timesteps")
    with pytest.raises(ValueError, match="comma-separated integers"):
        _parse_ints("0,bad", "timesteps")


def test_paired_bootstrap_reports_known_arm_difference():
    records = []
    for run, value in zip(RUN_LABELS, (1.0, 3.0)):
        for index in range(4):
            records.append(
                {
                    "run": run,
                    "test_set": "geneval_gs1",
                    "epoch": COMMON_EPOCHS[-1],
                    "index": index,
                    "pixel_rmse": value,
                    "lpips": value,
                    "clip_cosine": value,
                    "dino_cosine": value,
                }
            )
    result = _paired_bootstrap_epoch80(records, repetitions=100)
    for metric in ("pixel_rmse", "lpips", "clip_cosine", "dino_cosine"):
        assert result[metric]["mean_32b_minus_9b"] == pytest.approx(2.0)
        assert result[metric]["ci95_low"] == pytest.approx(2.0)
        assert result[metric]["ci95_high"] == pytest.approx(2.0)


def test_cka_and_mmd_identical_features():
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert _linear_cka(features, features) == pytest.approx(1.0)
    assert _rbf_mmd(features, features) == pytest.approx(0.0)
