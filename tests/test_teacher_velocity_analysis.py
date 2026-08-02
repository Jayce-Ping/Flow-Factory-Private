import numpy as np
import pytest

from scripts.xopd_analysis.analyze_teacher_velocity_fields import (
    _cluster_bootstrap_ci,
    _detail_alignment,
    _spatial_metrics,
)


def _packed_grid() -> np.ndarray:
    y, x = np.mgrid[0:32, 0:32]
    grid = (x + 2 * y).astype(np.float32)[..., None]
    return np.repeat(grid, 4, axis=-1).reshape(1, 1024, 4)


def test_detail_alignment_detects_reinforcement_and_smoothing():
    base = _packed_grid()
    positive = _detail_alignment(base, base)
    negative = _detail_alignment(base, -base)
    assert positive["gradient_cosine"] == pytest.approx(1.0)
    assert positive["laplacian_cosine"] == pytest.approx(1.0)
    assert negative["gradient_cosine"] == pytest.approx(-1.0)
    assert negative["laplacian_cosine"] == pytest.approx(-1.0)


def test_spatial_metrics_report_more_detail_for_checkerboard():
    smooth = np.zeros((1, 1024, 4), dtype=np.float32)
    checker = np.indices((32, 32)).sum(axis=0) % 2
    checker = np.repeat(checker[..., None], 4, axis=-1).astype(np.float32)
    checker = checker.reshape(1, 1024, 4)
    smooth_metrics = _spatial_metrics(smooth)
    checker_metrics = _spatial_metrics(checker)
    assert checker_metrics["high_frequency_ratio"] > smooth_metrics["high_frequency_ratio"]
    assert checker_metrics["tv_rms"] > smooth_metrics["tv_rms"]


def test_cluster_bootstrap_resamples_prompts_not_steps():
    rows = [
        {"sample_index": sample, "metric": float(sample)}
        for sample in range(4)
        for _ in range(3)
    ]
    result = _cluster_bootstrap_ci(rows, "metric", repetitions=1000)
    assert result["mean"] == pytest.approx(1.5)
    assert result["sample_clusters"] == 4
    assert result["observations"] == 12
