import numpy as np
import pytest
import torch

from scripts.xopd_analysis.analyze_full_capture_x0 import (
    cluster_bootstrap,
    compute_x0,
    image_sharpness,
    parse_steps,
    reconstruct_and_validate_latent_ids,
)


def test_reconstructed_flux2_ids_match_512px_row_major_layout():
    ids = reconstruct_and_validate_latent_ids(1024, 128)
    assert ids.shape == (1, 1024, 4)
    torch.testing.assert_close(ids[0, 0], torch.tensor([0, 0, 0, 0]))
    torch.testing.assert_close(ids[0, 31], torch.tensor([0, 0, 31, 0]))
    torch.testing.assert_close(ids[0, 32], torch.tensor([0, 1, 0, 0]))
    torch.testing.assert_close(ids[0, -1], torch.tensor([0, 31, 31, 0]))


def test_compute_x0_uses_rectified_flow_formula_and_validates_inputs():
    x_t = np.full((1, 4, 2), 2.0, dtype=np.float16)
    velocity = np.full((1, 4, 2), 3.0, dtype=np.float16)
    np.testing.assert_array_equal(compute_x0(x_t, velocity, 500.0), 0.5)
    with pytest.raises(ValueError, match="matching x_t/velocity shapes"):
        compute_x0(x_t, velocity[:, :-1], 500.0)
    with pytest.raises(ValueError, match=r"timestep in \[0,1000\]"):
        compute_x0(x_t, velocity, 1001.0)


def test_image_sharpness_detects_checkerboard_detail():
    flat = torch.full((3, 512, 512), 0.5)
    checker = (
        (torch.arange(512)[:, None] + torch.arange(512)[None, :]) % 2
    ).float()
    checker = checker.unsqueeze(0).expand(3, -1, -1)
    flat_metrics = image_sharpness(flat)
    checker_metrics = image_sharpness(checker)
    for metric in ("laplacian_variance", "gradient_energy", "fft_high_frequency"):
        assert checker_metrics[metric] > flat_metrics[metric]


def test_cluster_bootstrap_keeps_prompt_steps_in_same_cluster():
    rows = [
        {"sample_index": 0, "metric": 0.0},
        {"sample_index": 0, "metric": 2.0},
        {"sample_index": 1, "metric": 10.0},
        {"sample_index": 1, "metric": 12.0},
    ]
    result = cluster_bootstrap(rows, "metric", repetitions=2000, seed=7)
    assert result["mean"] == pytest.approx(6.0)
    assert result["prompt_clusters"] == 2
    assert result["observations"] == 4


def test_parse_steps_rejects_duplicates_and_nonintegers():
    assert parse_steps("0,4,27") == (0, 4, 27)
    with pytest.raises(ValueError, match="unique steps"):
        parse_steps("0,0")
    with pytest.raises(ValueError, match="comma-separated integer"):
        parse_steps("0,bad")
