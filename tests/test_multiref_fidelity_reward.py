import torch

from flow_factory.rewards.multiref_fidelity import aggregate_reference_scores
from flow_factory.rewards.registry import get_reward_model_class


def test_aggregation_exposes_tail_and_coverage() -> None:
    metrics = aggregate_reference_scores(
        [
            torch.tensor([0.9, 0.8, 0.1]),
            torch.tensor([0.5, 0.5]),
        ],
        coverage_threshold=0.25,
    )
    torch.testing.assert_close(metrics["mean"], torch.tensor([0.6, 0.5]))
    torch.testing.assert_close(metrics["min"], torch.tensor([0.1, 0.5]))
    torch.testing.assert_close(metrics["p10"], torch.tensor([0.24, 0.5]))
    torch.testing.assert_close(metrics["coverage"], torch.tensor([2 / 3, 1.0]))
    torch.testing.assert_close(metrics["reference_count"], torch.tensor([3.0, 2.0]))


def test_registry_exposes_multiref_fidelity() -> None:
    cls = get_reward_model_class("multiref_fidelity")
    assert cls.__name__ == "MultiReferenceFidelityRewardModel"
