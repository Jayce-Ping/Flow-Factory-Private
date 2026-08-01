"""Per-reference CLIP fidelity with minimum/p10/coverage aggregation."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments


class MultiReferenceFidelityRewardModel(PointwiseRewardModel):
    required_fields = ("image", "condition_images")
    use_tensor_inputs = False
    DEFAULT_MODEL = "openai/clip-vit-large-patch14"

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        model_path = config.extra_kwargs.get("model_name_or_path", self.DEFAULT_MODEL)
        self.model = CLIPModel.from_pretrained(model_path, torch_dtype=self.dtype)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.to(self.device).eval()
        self.aggregation = config.extra_kwargs.get("aggregation", "p10")
        if self.aggregation not in ("mean", "min", "p10"):
            raise ValueError(
                f"expected aggregation in ('mean','min','p10'), got {self.aggregation!r}"
            )
        self.coverage_threshold = float(
            config.extra_kwargs.get("coverage_threshold", 0.25)
        )

    @torch.no_grad()
    def __call__(
        self,
        image: List[Image.Image],
        condition_images: List[List[Image.Image]],
        **kwargs,
    ) -> RewardModelOutput:
        if len(image) != len(condition_images):
            raise ValueError(
                f"expected equal generated/reference batch sizes, got "
                f"{len(image)} and {len(condition_images)}"
            )
        if any(not references for references in condition_images):
            empty = [index for index, references in enumerate(condition_images) if not references]
            raise ValueError(f"multi-reference reward received empty reference lists at {empty}")
        generated = self._encode_images(image)
        flat_references = [reference for references in condition_images for reference in references]
        reference_embeds = self._encode_images(flat_references)

        per_sample: list[torch.Tensor] = []
        offset = 0
        for sample_index, references in enumerate(condition_images):
            count = len(references)
            current = reference_embeds[offset : offset + count]
            per_sample.append(current @ generated[sample_index])
            offset += count
        metrics = aggregate_reference_scores(
            per_sample,
            coverage_threshold=self.coverage_threshold,
        )
        rewards = metrics[self.aggregation]
        return RewardModelOutput(
            rewards=rewards.float().cpu(),
            extra_info={key: value.float().cpu() for key, value in metrics.items()},
        )

    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        features = self.model.get_image_features(pixel_values=pixel_values)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        return F.normalize(features.float(), p=2, dim=-1)


def aggregate_reference_scores(
    per_sample: list[torch.Tensor],
    *,
    coverage_threshold: float,
) -> dict[str, torch.Tensor]:
    if not per_sample:
        raise ValueError("expected at least one sample of reference scores")
    if not 0 <= coverage_threshold <= 1:
        raise ValueError(
            f"expected coverage_threshold in [0,1], got {coverage_threshold!r}"
        )
    for index, values in enumerate(per_sample):
        if values.ndim != 1 or values.numel() < 1:
            raise ValueError(
                f"expected non-empty 1D reference scores for sample {index}, "
                f"got shape={tuple(values.shape)}"
            )
        if not torch.isfinite(values).all():
            raise ValueError(
                f"expected finite reference scores for sample {index}, got {values.tolist()}"
            )
    return {
        "mean": torch.stack([values.mean() for values in per_sample]),
        "min": torch.stack([values.min() for values in per_sample]),
        "p10": torch.stack(
            [torch.quantile(values.float(), 0.10) for values in per_sample]
        ),
        "coverage": torch.stack(
            [(values >= coverage_threshold).float().mean() for values in per_sample]
        ),
        "reference_count": torch.tensor(
            [values.numel() for values in per_sample],
            device=per_sample[0].device,
            dtype=torch.float32,
        ),
    }
