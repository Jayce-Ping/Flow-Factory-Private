# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Score-centered GRPO: closed-form centering, scheduler convention, and loss wiring."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest
import torch

from flow_factory.hparams import Arguments
from flow_factory.hparams.training_args import SCGRPOTrainingArguments, get_training_args_class
from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import (
    BaseSample,
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    ReplayStep,
)
from flow_factory.scheduler import SchedulerGroup, SDESchedulerOutput
from flow_factory.scheduler.flow_match_euler_discrete import FlowMatchEulerDiscreteSDEScheduler
from flow_factory.scheduler.minimax_h3 import MiniMaxH3SDEScheduler
from flow_factory.trainers.registry import get_trainer_class
from flow_factory.trainers.rl.sc_grpo import SCGRPOTrainer

ROOT = Path(__file__).resolve().parents[2]
SDE_DYNAMICS = ("Flow-SDE", "Dance-SDE", "CPS")


class TrainingArgsFake(dict):
    """Mapping/attribute hybrid mirroring ``ArgABC`` unpacking behaviour."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


class SchedulerFake:
    """Scheduler-like object exposing only the dynamics the trainer reads."""

    def __init__(self, dynamics_type: str = "Flow-SDE") -> None:
        self.dynamics_type = dynamics_type
        self.noise_level = 0.7
        self.train_timesteps = torch.tensor([0])

    def step(self) -> None:
        """Provide scheduler compatibility."""


class AdapterFake(BaseAdapter):
    """Minimal concrete adapter; the trainer only uses its reduction contract."""

    def load_pipeline(self) -> Any:
        """Return an unused pipeline fake."""
        raise NotImplementedError

    def decode_latents(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return latents unchanged."""
        return latents

    def inference(self, **kwargs: Any) -> List[BaseSample]:
        """Return no samples."""
        return []

    def forward(self, **kwargs: Any) -> SDESchedulerOutput:
        """Reject accidental real forwards."""
        raise NotImplementedError


class StructuredAdapterFake(AdapterFake):
    """Adapter fake declaring a video/audio component contract."""

    trajectory_component_order = ("video", "audio")


def _adapter(dynamics_type: str = "Flow-SDE") -> AdapterFake:
    adapter = object.__new__(AdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake(dynamics_type))
    adapter.scheduler_group = adapter.build_scheduler_group()
    return adapter


def _structured_adapter(video_dynamics: str, audio_dynamics: str) -> StructuredAdapterFake:
    adapter = object.__new__(StructuredAdapterFake)
    video = SchedulerFake(video_dynamics)
    adapter.pipeline = SimpleNamespace(scheduler=video)
    adapter.scheduler_group = SchedulerGroup(
        {"video": video, "audio": SchedulerFake(audio_dynamics)},
        primary_name="video",
    )
    return adapter


def _trainer(adapter: BaseAdapter, **training_args: Any) -> SCGRPOTrainer:
    trainer = object.__new__(SCGRPOTrainer)
    trainer.adapter = adapter
    trainer.training_args = TrainingArgsFake(training_args)
    return trainer


def _replay(
    state: Dict[str, torch.Tensor],
    next_state: Dict[str, torch.Tensor],
    log_prob: Optional[torch.Tensor] = None,
) -> ReplayStep:
    names = tuple(state)
    batch_size = state[names[0]].shape[0]
    return ReplayStep(
        state=LatentState(dict(state)),
        next_state=LatentState(dict(next_state)),
        times=ComponentTimes(
            timestep={name: torch.full((batch_size,), 500.0) for name in names},
            next_timestep={name: torch.zeros(batch_size) for name in names},
        ),
        log_prob=log_prob,
        component_log_probs=None if log_prob is None else {name: log_prob for name in names},
    )


def _policy_output(step: SDESchedulerOutput) -> MultiModalStepOutput:
    return MultiModalStepOutput(
        next_state_mean=LatentState({"latent": step.next_latents_mean}),
        std_dev_t={"latent": step.std_dev_t},
        dt={"latent": step.dt},
        log_prob=step.log_prob,
        component_log_probs={"latent": step.log_prob},
    )


def _euler_step(dynamics_type: str) -> Callable[..., SDESchedulerOutput]:
    scheduler = FlowMatchEulerDiscreteSDEScheduler(dynamics_type=dynamics_type)

    def step(velocity: torch.Tensor, latents: torch.Tensor, **kwargs: Any) -> SDESchedulerOutput:
        return scheduler.step(
            velocity=velocity,
            timestep=torch.tensor(750.0),
            timestep_next=torch.tensor(500.0),
            latents=latents,
            noise_level=0.7,
            **kwargs,
        )

    return step


def _minimax_step(dynamics_type: str) -> Callable[..., SDESchedulerOutput]:
    scheduler = MiniMaxH3SDEScheduler(dynamics_type=dynamics_type, noise_level=0.7)
    scheduler.set_timesteps(4)
    # Explicit coordinates keep repeated replays of one transition off the stateful step index.
    timestep, sigma, sigma_next = scheduler.timesteps[1], scheduler.sigmas[1], scheduler.sigmas[2]

    def step(velocity: torch.Tensor, latents: torch.Tensor, **kwargs: Any) -> SDESchedulerOutput:
        return scheduler.step(
            velocity=velocity,
            timestep=timestep,
            latents=latents,
            sigma=sigma,
            sigma_next=sigma_next,
            **kwargs,
        )

    return step


@pytest.mark.parametrize("make_step", [_euler_step, _minimax_step], ids=["euler", "minimax_h3"])
@pytest.mark.parametrize("dynamics_type", SDE_DYNAMICS)
def test_sampler_kl_is_the_negative_rollout_expected_log_prob(
    make_step: Callable[[str], Callable[..., SDESchedulerOutput]], dynamics_type: str
) -> None:
    """``KL(q || p)`` must equal ``log p(mu_p) - log p(mu_q)`` in the scheduler's own convention.

    The Gaussian identity ``E_q[log p(X)] = log p(mu_q) + const`` makes this the exact
    centering term; CPS uses an unnormalized ``log_prob`` and must match it too.
    """
    torch.manual_seed(0)
    step = make_step(dynamics_type)
    latents = torch.randn(3, 2, 5)
    rollout_velocity = torch.randn(3, 2, 5)
    policy_velocity = rollout_velocity + 0.3 * torch.randn(3, 2, 5)

    rollout = step(rollout_velocity, latents, generator=torch.Generator().manual_seed(1))
    at_sample = step(policy_velocity, latents, next_latents=rollout.next_latents)
    at_rollout_mean = step(policy_velocity, latents, next_latents=rollout.next_latents_mean)
    at_policy_mean = step(policy_velocity, latents, next_latents=at_sample.next_latents_mean)

    trainer = _trainer(_adapter(dynamics_type))
    replay = _replay({"latent": latents}, {"latent": rollout.next_latents})
    sampler_kl = trainer._sampler_kl(
        _policy_output(at_sample), replay, LatentState({"latent": rollout.next_latents_mean})
    )

    torch.testing.assert_close(
        sampler_kl, at_policy_mean.log_prob - at_rollout_mean.log_prob, rtol=1e-5, atol=1e-6
    )
    assert bool((sampler_kl > 0).all())


def test_score_centering_cancels_drift_under_a_constant_advantage() -> None:
    """With a constant advantage, the centered expected gradient vanishes; plain REINFORCE drifts."""
    torch.manual_seed(0)
    batch_size, channels = 40000, 4
    step = _euler_step("Flow-SDE")
    latents = torch.randn(1, channels).expand(batch_size, channels)
    base_velocity = torch.randn(1, channels).expand(batch_size, channels)
    # Large enough that the plain drift dominates the Monte Carlo noise of 4e4 rollouts.
    mismatch = torch.full((channels,), 1.0)

    with torch.no_grad():
        rollout = step(
            base_velocity + mismatch, latents, generator=torch.Generator().manual_seed(2)
        )
    trainer = _trainer(_adapter("Flow-SDE"))
    replay = _replay({"latent": latents}, {"latent": rollout.next_latents})
    sampler_mean = LatentState({"latent": rollout.next_latents_mean})

    gradients = {}
    for score_centering in (False, True):
        policy_shift = torch.zeros(channels, requires_grad=True)
        policy = step(base_velocity + policy_shift, latents, next_latents=rollout.next_latents)
        objective = policy.log_prob
        if score_centering:
            objective = objective + trainer._sampler_kl(
                _policy_output(policy), replay, sampler_mean
            )
        torch.mean(-1.0 * objective).backward()
        gradients[score_centering] = policy_shift.grad

    plain_norm = gradients[False].norm()
    centered_norm = gradients[True].norm()
    assert plain_norm > 0
    assert centered_norm < 0.05 * plain_norm


def test_sampler_kl_weights_components_by_element_count_and_dynamics() -> None:
    """Video (Flow-SDE) and audio (CPS) KLs combine element-weighted, each in its own convention."""
    torch.manual_seed(4)
    policy = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    rollout = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    std_dev_t = {
        "video": torch.tensor([0.3, 0.32]).reshape(2, 1, 1),
        "audio": torch.tensor([0.6, 0.62]).reshape(2, 1),
    }
    dt = {
        "video": torch.tensor([-0.4, -0.42]).reshape(2, 1, 1),
        "audio": torch.tensor([-0.2, -0.22]).reshape(2, 1),
    }
    trainer = _trainer(_structured_adapter("Flow-SDE", "CPS"))
    replay = _replay(
        {"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)},
        {"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)},
    )
    output = MultiModalStepOutput(
        next_state_mean=LatentState(dict(policy)), std_dev_t=std_dev_t, dt=dt
    )

    sampler_kl = trainer._sampler_kl(output, replay, LatentState(dict(rollout)))

    video_scale = std_dev_t["video"] * torch.sqrt(-dt["video"])
    video_sum = ((rollout["video"] - policy["video"]) ** 2 / (2 * video_scale**2)).flatten(1)
    audio_sum = ((rollout["audio"] - policy["audio"]) ** 2).flatten(1)
    expected = (video_sum.sum(dim=1) + audio_sum.sum(dim=1)) / 17
    torch.testing.assert_close(sampler_kl, expected)


def _single_output(
    std_dev_t: torch.Tensor, dt: torch.Tensor, mean: Optional[torch.Tensor] = None
) -> MultiModalStepOutput:
    return MultiModalStepOutput(
        next_state_mean=LatentState({"latent": torch.zeros(1, 4) if mean is None else mean}),
        std_dev_t={"latent": std_dev_t},
        dt={"latent": dt},
    )


def test_sampler_kl_rejects_ode_dynamics() -> None:
    trainer = _trainer(_adapter("ODE"))
    replay = _replay({"latent": torch.zeros(1, 4)}, {"latent": torch.zeros(1, 4)})
    with pytest.raises(ValueError, match=r"SC-GRPO sampler KL.*'ODE'.*constraints #7"):
        trainer._sampler_kl(
            _single_output(torch.full((1, 1), 0.3), torch.full((1, 1), -0.5)),
            replay,
            LatentState({"latent": torch.zeros(1, 4)}),
        )


def test_sampler_kl_rejects_a_zero_stochastic_transition_scale() -> None:
    trainer = _trainer(_adapter("CPS"))
    replay = _replay({"latent": torch.zeros(1, 4)}, {"latent": torch.zeros(1, 4)})
    with pytest.raises(ValueError, match=r"SC-GRPO sampler KL.*strictly positive.*CPS"):
        trainer._sampler_kl(
            _single_output(torch.zeros(1, 1), torch.full((1, 1), -0.5)),
            replay,
            LatentState({"latent": torch.zeros(1, 4)}),
        )


def test_sampler_kl_rejects_a_rollout_mean_with_a_different_shape() -> None:
    trainer = _trainer(_adapter())
    replay = _replay({"latent": torch.zeros(1, 4)}, {"latent": torch.zeros(1, 4)})
    with pytest.raises(ValueError, match=r"next_latents_mean for component 'latent'.*\(1, 4\)"):
        trainer._sampler_kl(
            _single_output(torch.full((1, 1), 0.3), torch.full((1, 1), -0.5)),
            replay,
            LatentState({"latent": torch.zeros(1, 3)}),
        )


def test_sampler_kl_rejects_a_rollout_mean_in_the_wrong_component_order() -> None:
    trainer = _trainer(_structured_adapter("Flow-SDE", "CPS"))
    state = {"video": torch.zeros(1, 4), "audio": torch.zeros(1, 2)}
    replay = _replay(state, state)
    output = MultiModalStepOutput(
        next_state_mean=LatentState(dict(state)),
        std_dev_t={"video": torch.full((1, 1), 0.3), "audio": torch.full((1, 1), 0.3)},
        dt={"video": torch.full((1, 1), -0.5), "audio": torch.full((1, 1), -0.5)},
    )
    with pytest.raises(ValueError, match=r"component order \('video', 'audio'\)"):
        trainer._sampler_kl(
            output,
            replay,
            LatentState({"audio": torch.zeros(1, 2), "video": torch.zeros(1, 4)}),
        )


class _AcceleratorFake:
    sync_gradients = False

    @staticmethod
    def backward(loss: torch.Tensor) -> None:
        loss.backward()


def _optimize_harness(score_centering: bool) -> tuple:
    """Run ``optimize()`` for one micro-batch and one timestep on a linear policy mean."""
    torch.manual_seed(7)
    batch_size, channels = 3, 4
    scale = torch.full((batch_size, 1), 0.2)
    dt = torch.full((batch_size, 1), -0.25)
    std_dev_t = scale / torch.sqrt(-dt)
    rollout_mean = torch.randn(batch_size, channels)
    sampled = rollout_mean + scale * torch.randn(batch_size, channels)
    base_mean = rollout_mean + 0.1 * torch.randn(batch_size, channels)
    advantage = torch.tensor([1.5, -0.5, 0.25])
    old_log_prob = torch.randn(batch_size)
    policy_shift = torch.zeros(channels, requires_grad=True)

    def policy_terms() -> tuple:
        mean = base_mean + policy_shift
        log_prob = (-((sampled - mean) ** 2) / (2 * scale**2)).mean(dim=1)
        return mean, log_prob

    adapter = _adapter("Flow-SDE")
    adapter.get_train_step_indices = lambda: torch.tensor([0])
    adapter.get_replay_step = lambda batch, index: _replay(
        {"latent": torch.zeros(batch_size, channels)}, {"latent": sampled}, old_log_prob
    )
    adapter.get_replay_callback = lambda batch, index, field: LatentState({"latent": rollout_mean})
    adapter.train = lambda: None

    def replay_forward(batch: Any, replay: ReplayStep, fields: tuple) -> MultiModalStepOutput:
        assert fields == ("log_prob", "next_latents_mean", "std_dev_t", "dt")
        mean, log_prob = policy_terms()
        return MultiModalStepOutput(
            next_state_mean=LatentState({"latent": mean}),
            std_dev_t={"latent": std_dev_t},
            dt={"latent": dt},
            log_prob=log_prob,
            component_log_probs={"latent": log_prob},
        )

    def iter_batches(samples: List[Any], per_device_batch_size: int) -> Iterator[Dict]:
        yield {"advantage": advantage}

    trainer = _trainer(
        adapter,
        per_device_batch_size=batch_size,
        num_inner_epochs=1,
        score_centering=score_centering,
        adv_clip_range=(-1.0, 1.0),
        kl_beta=0.0,
        kl_type="x-based",
        kl_guidance_scale=None,
    )
    trainer.accelerator = _AcceleratorFake()
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.autocast = nullcontext
    trainer.accumulate_gradients = nullcontext
    trainer.epoch = 0
    trainer._order_samples_for_optimize = lambda samples, inner_epoch: samples
    trainer._iter_prefetched_batches = iter_batches
    trainer._replay_forward = replay_forward

    trainer.optimize([object()] * batch_size)

    mean, log_prob = policy_terms()
    clipped_advantage = advantage.clamp(-1.0, 1.0)
    objective = log_prob
    if score_centering:
        objective = objective + ((rollout_mean - mean) ** 2 / (2 * scale**2)).mean(dim=1)
    expected_grad = torch.autograd.grad(torch.mean(-clipped_advantage * objective), policy_shift)[0]
    return policy_shift.grad, expected_grad, sampled - rollout_mean, clipped_advantage, scale


@pytest.mark.parametrize("score_centering", [True, False], ids=["centered", "plain"])
def test_optimize_backpropagates_the_selected_objective(score_centering: bool) -> None:
    gradient, expected, _, _, _ = _optimize_harness(score_centering)
    torch.testing.assert_close(gradient, expected)


def test_optimize_centered_gradient_only_depends_on_the_rollout_noise() -> None:
    """``d/dmu [log p(x') + KL(q||p)] = (x' - mu_q) / (D s^2)``: no ``mu_q - mu_theta`` drift."""
    gradient, _, rollout_noise, advantage, scale = _optimize_harness(score_centering=True)
    channels = rollout_noise.shape[1]
    expected = -(advantage[:, None] * rollout_noise / (channels * scale**2)).mean(dim=0)
    torch.testing.assert_close(gradient, expected)


def test_training_arguments_resolve_and_validate() -> None:
    assert get_training_args_class("sc-grpo") is SCGRPOTrainingArguments
    assert get_trainer_class("sc-grpo") is SCGRPOTrainer
    assert SCGRPOTrainer.paradigm == "coupled"

    arguments = SCGRPOTrainingArguments(kl_beta="1e-3", kl_guidance_scale="4.5")
    assert arguments.score_centering is True
    assert arguments.kl_beta == pytest.approx(1e-3)
    assert arguments.get_preprocess_guidance_scale() == pytest.approx(
        max(arguments.guidance_scale, 4.5)
    )
    assert not hasattr(arguments, "clip_range")


def test_example_config_parses_into_sc_grpo_arguments() -> None:
    config = Arguments.load_from_yaml(str(ROOT / "examples/sc_grpo/lora/sd3_5/default.yaml"))

    assert isinstance(config.training_args, SCGRPOTrainingArguments)
    assert config.training_args.trainer_type == "sc-grpo"
    assert config.training_args.score_centering is True
    assert config.scheduler_args.dynamics_type == "Flow-SDE"


def test_training_arguments_reject_a_non_boolean_score_centering() -> None:
    with pytest.raises(TypeError, match=r"expected bool for score_centering, got str: 'yes'"):
        SCGRPOTrainingArguments(score_centering="yes")
