from __future__ import annotations

import tempfile
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from torch import nn

from flow_factory.hparams import RewardGuidanceDistillTrainingArguments
from flow_factory.models.stable_diffusion.reward_control import (
    CombinedTimestepRewardControlTextProjEmbeddings,
    RewardControlEmbedding,
)
from flow_factory.trainers.ensemble_eval.common import (
    cache_scheduler_step_signature,
)
from flow_factory.trainers.reward_guidance.control import (
    ControlStrengthSampler,
    compose_reward_residual_oracle,
    pseudo_huber_loss,
)
from flow_factory.trainers.reward_guidance.distill import (
    RewardGuidanceDistillTrainer,
)
from flow_factory.utils.lora_loader import (
    set_peft_state_allowing_missing_modules_to_save,
)


class _TimeProjection(nn.Module):
    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (timestep, timestep.square(), timestep.sin(), timestep.cos()),
            dim=-1,
        )


class _StockTimeTextEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.time_proj = _TimeProjection()
        self.timestep_embedder = nn.Linear(4, 6)
        self.text_embedder = nn.Linear(3, 6)

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        projected = self.time_proj(timestep)
        return self.timestep_embedder(projected.to(pooled_projection.dtype)) + self.text_embedder(
            pooled_projection
        )


def _training_args(
    teacher_names: list[str],
) -> RewardGuidanceDistillTrainingArguments:
    return RewardGuidanceDistillTrainingArguments(
        trainer_type="reward-guidance-distill",
        teachers=[
            {
                "name": name,
                "path": f"owner/{name}",
                "guidance_scale": 4.5,
            }
            for name in teacher_names
        ],
        control_ranges={name: [-0.5, 2.0] for name in teacher_names},
        guidance_scale=1.0,
        target_guidance_scale=4.5,
    )


@pytest.mark.parametrize("control_count", [1, 3, 5])
def test_control_sampler_is_deterministic_for_arbitrary_k(
    control_count: int,
) -> None:
    names = [f"teacher_{index}" for index in range(control_count)]
    sampler = ControlStrengthSampler(
        control_names=names,
        control_ranges={name: (-0.5, 2.0) for name in names},
        probabilities={
            "anchor": 0.1,
            "axis": 0.35,
            "sparse_joint": 0.25,
            "dense_joint": 0.3,
        },
    )
    kwargs = {
        "batch_size": 128,
        "base_seed": 17,
        "epoch": 3,
        "process_index": 2,
        "batch_index": 9,
        "device": "cpu",
    }
    first = sampler.sample(**kwargs)
    second = sampler.sample(**kwargs)
    assert torch.equal(first.values, second.values)
    assert first.strata == second.strata
    assert first.values.shape == (128, control_count)
    assert torch.all(first.values >= -0.5)
    assert torch.all(first.values <= 2.0)
    assert set(first.strata) <= {
        "anchor",
        "axis",
        "sparse_joint",
        "dense_joint",
    }


def test_reward_residual_oracle_preserves_direct_coefficients() -> None:
    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    teachers = torch.stack(
        (
            base + torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
            base + torch.tensor([[0.0, 3.0], [0.0, 4.0]]),
            base - 2.0,
        )
    )
    controls = torch.tensor([[2.0, 0.5, -0.5], [0.0, 1.0 / 3.0, 1.5]])
    actual = compose_reward_residual_oracle(base, teachers, controls)
    expected = base.float()
    for index in range(teachers.shape[0]):
        expected = expected + controls[:, index, None] * (teachers[index] - base)
    assert torch.equal(actual, expected)


def test_zero_control_is_exact_stock_embedding_and_context_cleans_up() -> None:
    torch.manual_seed(0)
    stock = _StockTimeTextEmbedding()
    extended = CombinedTimestepRewardControlTextProjEmbeddings(
        stock,
        control_names=("geneval", "ocr"),
        embedding_dim=6,
        fourier_dim=8,
        hidden_dim=12,
    )
    timestep = torch.tensor([0.1, 0.9])
    pooled = torch.randn(2, 3)
    expected = stock(timestep, pooled)

    with extended.use_reward_control(torch.zeros(2, 2)):
        actual = extended(timestep, pooled)
    assert torch.equal(actual, expected)
    assert extended._active_reward_control is None

    with (
        pytest.raises(RuntimeError, match="deliberate"),
        extended.use_reward_control(torch.ones(2, 2)),
    ):
        raise RuntimeError("deliberate")
    assert extended._active_reward_control is None


def test_control_anchor_remains_exact_after_training_parameters_change() -> None:
    embedding = RewardControlEmbedding(
        control_names=("geneval",),
        embedding_dim=6,
        fourier_dim=8,
        hidden_dim=12,
    )
    with torch.no_grad():
        embedding.embedders[0][-1].weight.normal_()
        embedding.embedders[0][-1].bias.normal_()
    zero = embedding(torch.zeros(4, 1), output_dtype=torch.float32)
    nonzero = embedding(torch.ones(4, 1), output_dtype=torch.float32)
    assert torch.equal(zero, torch.zeros_like(zero))
    assert not torch.equal(nonzero, torch.zeros_like(nonzero))


def test_pseudo_huber_is_finite_and_rejects_bad_delta() -> None:
    error = torch.tensor([0.0, 1.0, 1000.0])
    loss = pseudo_huber_loss(error, delta=0.01)
    assert torch.isfinite(loss).all()
    assert loss[0] == 0
    with pytest.raises(ValueError, match="positive"):
        pseudo_huber_loss(error, delta=0.0)


def test_training_args_enforce_matched_cfg_and_control_order() -> None:
    parsed = _training_args(["geneval", "pickscore", "ocr"])
    assert parsed.reward_control_names == ["geneval", "pickscore", "ocr"]
    with pytest.raises(ValueError, match="exactly match teacher order"):
        RewardGuidanceDistillTrainingArguments(
            trainer_type="reward-guidance-distill",
            teachers=[
                {
                    "name": "geneval",
                    "path": "owner/geneval",
                    "guidance_scale": 4.5,
                }
            ],
            reward_control_names=["ocr"],
            control_ranges={"geneval": [-0.5, 2.0]},
            guidance_scale=1.0,
        )
    with pytest.raises(ValueError, match="matched CFG"):
        RewardGuidanceDistillTrainingArguments(
            trainer_type="reward-guidance-distill",
            teachers=[
                {
                    "name": "geneval",
                    "path": "owner/geneval",
                    "guidance_scale": 4.0,
                }
            ],
            control_ranges={"geneval": [-0.5, 2.0]},
            guidance_scale=1.0,
            target_guidance_scale=4.5,
        )


class _ToyControlModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.control_embedder = RewardControlEmbedding(
            control_names=("geneval",),
            embedding_dim=4,
            fourier_dim=8,
            hidden_dim=8,
        )

    def forward(self, value: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return self.proj(value) + self.control_embedder(control, output_dtype=value.dtype)


def test_peft_modules_to_save_round_trip_preserves_control_embedder() -> None:
    config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["proj"],
        modules_to_save=["control_embedder"],
    )
    model = get_peft_model(_ToyControlModel(), config)
    control_module = model.base_model.model.control_embedder.modules_to_save["default"]
    with torch.no_grad():
        control_module.embedders[0][-1].weight.fill_(0.125)

    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        loaded = PeftModel.from_pretrained(_ToyControlModel(), directory)
        loaded_control = loaded.base_model.model.control_embedder.modules_to_save["default"]
        assert torch.equal(
            loaded_control.embedders[0][-1].weight,
            torch.full_like(loaded_control.embedders[0][-1].weight, 0.125),
        )


def test_teacher_lora_may_omit_student_only_control_module() -> None:
    teacher = get_peft_model(
        _ToyControlModel(),
        LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]),
    )
    student = get_peft_model(
        _ToyControlModel(),
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["proj"],
            modules_to_save=["control_embedder"],
        ),
    )
    set_peft_state_allowing_missing_modules_to_save(
        student,
        get_peft_model_state_dict(teacher),
        adapter_name="default",
        allowed_modules_to_save=("control_embedder",),
    )


def test_teacher_lora_rejects_real_rank_shape_mismatch() -> None:
    teacher = get_peft_model(
        _ToyControlModel(),
        LoraConfig(r=1, lora_alpha=2, target_modules=["proj"]),
    )
    student = get_peft_model(
        _ToyControlModel(),
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["proj"],
            modules_to_save=["control_embedder"],
        ),
    )
    with pytest.raises(ValueError, match="tensor shapes"):
        set_peft_state_allowing_missing_modules_to_save(
            student,
            get_peft_model_state_dict(teacher),
            adapter_name="default",
            allowed_modules_to_save=("control_embedder",),
        )


class _FakeScheduler:
    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        timestep_next: torch.Tensor | None = None,
        return_dict: bool = True,
        return_kwargs: list[str] | None = None,
        **_: object,
    ) -> SimpleNamespace:
        assert return_dict
        return SimpleNamespace(
            noise_pred=noise_pred,
            next_latents=latents - noise_pred,
        )


class _FakeOracleAdapter:
    def __init__(self) -> None:
        self.scheduler = _FakeScheduler()
        self.transformer = nn.Linear(1, 1)
        self._velocity = torch.tensor(0.0, requires_grad=True)

    def get_component_unwrapped(self, _: str) -> nn.Module:
        return self.transformer

    def get_component(self, _: str) -> nn.Module:
        return self.transformer

    def set_component(self, _: str, module: nn.Module) -> None:
        self.transformer = module

    @contextmanager
    def use_ref_parameters(self):
        previous = self._velocity
        self._velocity = torch.tensor(1.0, requires_grad=True)
        try:
            yield
        finally:
            self._velocity = previous

    @contextmanager
    def use_named_parameters(self, name: str):
        values = {"geneval": 3.0, "ocr": -1.0}
        previous = self._velocity
        self._velocity = torch.tensor(values[name], requires_grad=True)
        try:
            yield
        finally:
            self._velocity = previous

    def predict_velocity(self, latents: torch.Tensor, **_: object) -> torch.Tensor:
        return torch.ones_like(latents) * self._velocity

    def forward(self, **_: object) -> None:
        raise AssertionError("the oracle context should replace adapter.forward")


def test_oracle_context_uses_scheduler_and_never_tracks_teacher_gradients() -> None:
    trainer = object.__new__(RewardGuidanceDistillTrainer)
    trainer.adapter = _FakeOracleAdapter()
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.training_args = SimpleNamespace(target_guidance_scale=4.5)
    trainer.teacher_names = ("geneval", "ocr")
    trainer.control_names = ("geneval", "ocr")
    trainer._scheduler_step_signature_cache = cache_scheduler_step_signature(
        trainer.adapter.scheduler.step
    )
    controls = torch.tensor([[0.5, -0.25], [2.0, 1.0]])
    latents = torch.zeros(2, 1, 2, 2)
    kwargs = {
        "t": torch.tensor([1.0, 1.0]),
        "t_next": torch.tensor([0.5, 0.5]),
        "latents": latents,
        "prompt_embeds": torch.zeros(2, 1, 1),
        "pooled_prompt_embeds": torch.zeros(2, 1),
        "negative_prompt_embeds": torch.zeros(2, 1, 1),
        "negative_pooled_prompt_embeds": torch.zeros(2, 1),
        "compute_log_prob": False,
        "return_kwargs": ["noise_pred", "next_latents"],
    }
    with trainer._oracle_inference_context(controls):
        output = trainer.adapter.forward(**kwargs)
    expected = torch.tensor([2.5, 3.0]).view(2, 1, 1, 1).expand_as(latents)
    assert torch.equal(output.noise_pred, expected)
    assert not output.noise_pred.requires_grad
    with pytest.raises(AssertionError, match="should replace"):
        trainer.adapter.forward(**kwargs)
