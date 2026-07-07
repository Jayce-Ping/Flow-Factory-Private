"""Standalone GPU smoke for ModelBundle + RoutedComponentProxy under accelerate FSDP2.

Validates the OOM-fallback pattern (shard BOTH a trainable "student" and a frozen
"teacher" as ONE FSDP root, route each forward through the bundle via a proxy)
WITHOUT the 32B/4B weights — tiny modules that mimic the diffusers transformer API
(`.config`, `.cache_context`, `.transformer_blocks`, `_no_split_modules`).

Run (2 GPUs, distinct port so it never collides with a live training job):
  source /opt/conda/etc/profile.d/conda.sh && conda activate ff
  cd /root/Flow-Factory-Private
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file config/accelerate_configs/fsdp2.yaml \
    --num_processes 2 --num_machines 1 --main_process_port 29888 \
    tests/smoke_fsdp_bundle.py

Checks: (1) FSDP shards params of BOTH members (local numel < global); (2) proxy
attribute fall-through (.config/.cache_context); (3) routed forward for student &
teacher through the single FSDP root; (4) backward updates ONLY the student, teacher
stays frozen; (5) accelerator.accumulate(bundle) works. Prints SMOKE_PASS / SMOKE_FAIL.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import torch
import torch.nn as nn
from accelerate import Accelerator

from flow_factory.models.model_bundle import (
    ModelBundle,
    RoutedComponentProxy,
    unwrap_routed_component,
)


class TinyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class TinyTransformer(nn.Module):
    """Mimics a diffusers transformer enough to exercise the proxy + auto-wrap."""

    _no_split_modules = ["TinyBlock"]

    def __init__(self, dim: int = 256, depth: int = 6):
        super().__init__()
        self.config = SimpleNamespace(dim=dim, depth=depth, kind="tiny")
        self.transformer_blocks = nn.ModuleList([TinyBlock(dim) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)

    @contextlib.contextmanager
    def cache_context(self, name: str):
        # diffusers-style no-op cache context; proxy must delegate this through.
        yield

    def forward(self, hidden_states):
        for blk in self.transformer_blocks:
            hidden_states = blk(hidden_states)
        return self.norm_out(hidden_states)


def _local_numel(p: torch.nn.Parameter) -> int:
    """numel of the LOCAL shard (DTensor for FSDP2, else full)."""
    t = getattr(p, "to_local", None)
    return p.to_local().numel() if callable(t) else p.numel()


def main():
    acc = Accelerator()
    dev = acc.device
    dim, depth, bs, seq = 256, 6, 2, 16

    student = TinyTransformer(dim, depth)
    teacher = TinyTransformer(dim, depth)
    teacher.requires_grad_(False)
    teacher.eval()

    student_full = sum(p.numel() for p in student.parameters())
    teacher_full = sum(p.numel() for p in teacher.parameters())

    bundle = ModelBundle({"student": student, "teacher": teacher})
    acc.print(f"[smoke] bundle._no_split_modules = {bundle._no_split_modules}")

    # Optimizer from student trainable params (mirrors trainer: created pre-prepare).
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-3)
    bundle, opt = acc.prepare(bundle, opt)

    # Recover members off the prepared root for proxy attribute delegation.
    root = unwrap_routed_component(bundle)
    inner_student = acc.unwrap_model(bundle).members["student"]
    inner_teacher = acc.unwrap_model(bundle).members["teacher"]
    student_proxy = RoutedComponentProxy(bundle, "student", inner_student)
    teacher_proxy = RoutedComponentProxy(bundle, "teacher", inner_teacher)

    checks = {}

    # (1) sharding: local shard smaller than the global param on >1 process
    s_local = sum(_local_numel(p) for p in inner_student.parameters())
    t_local = sum(_local_numel(p) for p in inner_teacher.parameters())
    if acc.num_processes > 1:
        checks["shard_student"] = s_local < student_full
        checks["shard_teacher"] = t_local < teacher_full
    else:
        checks["shard_student"] = checks["shard_teacher"] = True  # 1-proc: no sharding expected
    acc.print(f"[smoke] student numel: global={student_full} local={s_local} | "
              f"teacher: global={teacher_full} local={t_local}")

    # (2) proxy attribute fall-through
    checks["proxy_config"] = getattr(student_proxy, "config").kind == "tiny"
    with student_proxy.cache_context("cond"):
        pass
    checks["proxy_cache_ctx"] = True

    x = torch.randn(bs, seq, dim, device=dev)

    # (3) routed student forward through the single FSDP root + (4) backward on student only
    with acc.accumulate(bundle):
        out = student_proxy(hidden_states=x)
        checks["student_fwd_shape"] = tuple(out.shape) == (bs, seq, dim)
        loss = out.float().pow(2).mean()
        acc.backward(loss)
    s_has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in inner_student.parameters())
    t_has_grad = any(p.grad is not None for p in inner_teacher.parameters())
    checks["student_grad"] = s_has_grad
    checks["teacher_no_grad"] = not t_has_grad
    opt.step(); opt.zero_grad()
    checks["accumulate_ctx"] = True

    # (3b) routed teacher forward through the same root, inference-only
    with torch.no_grad():
        t_out = teacher_proxy(hidden_states=x)
    checks["teacher_fwd_shape"] = tuple(t_out.shape) == (bs, seq, dim)

    ok = all(checks.values())
    acc.wait_for_everyone()
    if acc.is_main_process:
        print("[smoke] checks:")
        for k, v in checks.items():
            print(f"    {'OK ' if v else 'FAIL'}  {k}")
        print("SMOKE_PASS" if ok else "SMOKE_FAIL")


if __name__ == "__main__":
    main()
