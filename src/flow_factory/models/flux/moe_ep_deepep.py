# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DeepEP backend for the weight-space MoE expert-parallel all-to-all.

Drop-in replacement for the NCCL ``preprocess`` / ``token_pre_all2all`` / ``tokens_post_all2all``
in ``moe_ep_comm.py`` (same signatures, so ``MoEFeedForward._ep_forward`` is backend-agnostic),
backed by DeepSeek's DeepEP (`deep_ep.Buffer`). Adapted from gen-ar's
``hy_parallelism.models.modules.moe.moe_parallel_deepep`` (grouped-GEMM dep removed; the caller
runs a per-expert ``nn.Module`` loop over the returned expert-contiguous tokens instead).

Intra-node EP (ep_size == gpus_per_node): the EP group is one node, so ``Buffer`` uses the
high-throughput **intranode NVLink** dispatch/combine kernels (no NVSHMEM/RDMA), which are the
fast, robust path here (validated by deepep_probe). Cross-node stays NCCL (the EDP grad sync).

deep_ep is only imported lazily (this module is only used when moe_ep_backend='deepep'), so the
plain NCCL EP path never requires deep_ep to be installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist

try:
    from deep_ep import Buffer
except ImportError as e:  # pragma: no cover - only hit when deepep backend requested w/o deep_ep
    raise ImportError(
        "moe_ep_backend='deepep' requires the deep_ep package (install into a torch-2.7.x / py3.12 "
        "env, e.g. the 'ff-new' env cloned from torch-base). Falling back to moe_ep_backend='nccl' "
        "avoids this dependency."
    ) from e


_buffer: Optional[Buffer] = None


@dataclass
class _DispatchState:
    handle: tuple
    sort_indices: torch.Tensor
    sorted_scores: torch.Tensor
    num_recv: int
    input_dtype: torch.dtype


_state_cache: dict[int, _DispatchState] = {}
_cache_counter: int = 0


def _next_cache_id() -> int:
    global _cache_counter
    _cache_counter += 1
    return _cache_counter


def _hidden_bytes(x: torch.Tensor) -> int:
    return x.size(-1) * max(x.element_size(), 2)


def _get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for cfg in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def _topk_idx_from_expert_mask(expert_mask: torch.Tensor) -> torch.Tensor:
    # [num_experts, topk, num_tokens] -> [num_tokens, topk] (global expert ids, int64 as DeepEP needs)
    return expert_mask.permute(2, 1, 0).long().argmax(dim=-1)


def _indices_to_multihot(
    indices: torch.Tensor, scores: torch.Tensor, num_local_experts: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = indices.shape[0]
    routing_map = torch.zeros((num_tokens, num_local_experts), dtype=torch.long, device=indices.device)
    weight_map = torch.zeros((num_tokens, num_local_experts), dtype=scores.dtype, device=indices.device)
    mask = indices != -1
    valid = indices[mask]
    rows = torch.arange(num_tokens, device=indices.device).repeat_interleave(mask.sum(dim=1))
    routing_map[rows, valid] = 1
    weight_map[rows, valid] = scores[mask]
    return routing_map.bool(), weight_map


def _permute_by_expert(
    tokens: torch.Tensor, routing_map: torch.Tensor, scores: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort received tokens so local-expert-0 rows come first, then expert-1, etc."""
    num_tokens, num_local_experts = routing_map.shape
    rmap_t = routing_map.bool().T.contiguous()
    token_idx = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_local_experts, -1)
    sort_indices = token_idx.masked_select(rmap_t)
    sorted_tokens = tokens.index_select(0, sort_indices)
    sorted_scores = scores.T.contiguous().masked_select(rmap_t)
    return sorted_tokens, sorted_scores, sort_indices


def _unpermute_by_expert(
    sorted_tokens: torch.Tensor, sort_indices: torch.Tensor, num_recv: int
) -> torch.Tensor:
    hidden = sorted_tokens.shape[-1]
    out = torch.zeros((num_recv, hidden), dtype=sorted_tokens.dtype, device=sorted_tokens.device)
    out.scatter_add_(0, sort_indices.unsqueeze(1).expand(-1, hidden), sorted_tokens)
    return out


class _Dispatch(torch.autograd.Function):
    """fwd: buffer.dispatch, bwd: buffer.combine."""

    @staticmethod
    def forward(ctx, x, topk_idx, topk_weights, num_experts, cache_id, group):
        buffer = _get_buffer(group, _hidden_bytes(x))
        (num_per_rank, num_per_rdma, num_per_expert, is_in_rank, _ev) = buffer.get_dispatch_layout(
            topk_idx=topk_idx, num_experts=num_experts
        )
        recv_x, recv_idx, recv_scores, num_recv_list, handle, _ev2 = buffer.dispatch(
            x=x, topk_idx=topk_idx, topk_weights=topk_weights.float(),
            num_tokens_per_rank=num_per_rank, num_tokens_per_rdma_rank=num_per_rdma,
            is_token_in_rank=is_in_rank, num_tokens_per_expert=num_per_expert,
            previous_event=None, async_finish=False, allocate_on_comm_stream=False,
        )
        ctx.handle = handle
        ctx.group = group
        ctx.input_dtype = x.dtype
        _state_cache[cache_id] = _DispatchState(
            handle=handle,
            sort_indices=torch.empty(0, dtype=torch.long, device=x.device),
            sorted_scores=torch.empty(0, device=x.device),
            num_recv=recv_x.shape[0],
            input_dtype=x.dtype,
        )
        num_recv = torch.tensor(num_recv_list, dtype=torch.int32, device="cpu")
        return recv_x, recv_idx, recv_scores, num_recv

    @staticmethod
    def backward(ctx, grad_recv_x, grad_recv_idx, grad_recv_scores, grad_num_recv):
        if grad_recv_x is None:
            return None, None, None, None, None, None
        buffer = _get_buffer(ctx.group, _hidden_bytes(grad_recv_x))
        grad_x, grad_scores, _ev = buffer.combine(
            x=grad_recv_x.contiguous().bfloat16(),  # DeepEP combine wants bf16
            handle=ctx.handle,
            topk_weights=grad_recv_scores.float() if grad_recv_scores is not None else None,
            previous_event=None, async_finish=False, allocate_on_comm_stream=False,
        )
        return (
            grad_x.to(ctx.input_dtype),
            None,
            grad_scores.to(ctx.input_dtype) if grad_scores is not None else None,
            None, None, None,
        )


class _Combine(torch.autograd.Function):
    """fwd: buffer.combine, bwd: buffer.dispatch."""

    @staticmethod
    def forward(ctx, x, handle, group):
        buffer = _get_buffer(group, _hidden_bytes(x))
        combined, _, _ev = buffer.combine(
            x=x.contiguous(), handle=handle,
            previous_event=None, async_finish=False, allocate_on_comm_stream=False,
        )
        ctx.handle = handle
        ctx.group = group
        return combined

    @staticmethod
    def backward(ctx, grad_combined):
        buffer = _get_buffer(ctx.group, _hidden_bytes(grad_combined))
        grad_x, *_rest, _ev = buffer.dispatch(
            x=grad_combined.contiguous(), handle=ctx.handle,
            previous_event=None, async_finish=False, allocate_on_comm_stream=False,
        )
        return grad_x, None, None


# ---------------------------------------------------------------------------
# public 3-function API (same signatures as moe_ep_comm's NCCL backend)
# ---------------------------------------------------------------------------
def preprocess(expert_mask: torch.Tensor, num_experts: int, ep_group: dist.ProcessGroup):
    """DeepEP resolves the token layout inside ``dispatch``; the only value the caller consumes is
    the per-local-expert token count (for the expert loop), which is known only AFTER dispatch. We
    return an empty counts buffer now and let ``token_pre_all2all`` fill it in-place. The 3rd and 4th
    return slots intentionally alias the SAME tensor (so the in-place fill is visible to the caller,
    which forwards the 3rd as ``num_global_tokens_per_local_expert`` and reads the 4th back as the
    counts). input/output splits are unused by DeepEP."""
    num_local_experts = num_experts // ep_group.size()
    counts = torch.empty(num_local_experts, dtype=torch.long, device=expert_mask.device)
    return None, None, counts, counts


def token_pre_all2all(
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
    input_splits,
    output_splits,
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_weights: Optional[torch.Tensor],
    ep_group: dist.ProcessGroup,
):
    """DeepEP dispatch: send each token to its top-k experts' owner ranks, regroup the received
    tokens contiguously per LOCAL expert, and fill ``num_global_tokens_per_local_expert`` in place.
    Returns ``(sorted_tokens, dispatched_routing_map, cache_id_tensor, org_shape)`` — the last three
    thread the DeepEP handle + sort state into :func:`tokens_post_all2all`."""
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim).contiguous()
    org_shape = hidden_states.shape

    topk_idx = _topk_idx_from_expert_mask(expert_mask)
    if routing_weights is not None:
        topk_weights = routing_weights.reshape(-1, topk_idx.shape[-1]).contiguous()
    else:
        topk_weights = torch.ones_like(topk_idx, dtype=torch.float32)

    num_local_experts = num_experts // ep_group.size()
    cache_id = _next_cache_id()
    recv_x, recv_idx, recv_scores, _num_recv = _Dispatch.apply(
        hidden_states, topk_idx, topk_weights, num_experts, cache_id, ep_group
    )

    dispatched_routing_map, dispatched_scores = _indices_to_multihot(
        recv_idx, recv_scores, num_local_experts
    )
    if num_global_tokens_per_local_expert is not None:
        num_global_tokens_per_local_expert.copy_(dispatched_routing_map.sum(dim=0))

    sorted_tokens, sorted_scores, sort_indices = _permute_by_expert(
        recv_x, dispatched_routing_map, dispatched_scores
    )
    state = _state_cache[cache_id]
    state.sort_indices = sort_indices
    state.sorted_scores = sorted_scores
    state.num_recv = recv_x.shape[0]

    cache_id_tensor = torch.tensor([cache_id], dtype=torch.int64, device="cpu")
    return sorted_tokens, dispatched_routing_map, cache_id_tensor, org_shape


def tokens_post_all2all(
    expert_outputs: torch.Tensor,
    selected_experts,
    num_experts: int,
    input_splits,
    output_splits,
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_map: torch.Tensor,
    local_input_permutation_mapping: torch.Tensor,
    org_hidden_states_shape: torch.Size,
    routing_weights: Optional[torch.Tensor],
    ep_group: dist.ProcessGroup,
):
    """DeepEP combine: weight each expert output by its (dispatch-time) top-k gate, unpermute back to
    the received order, then ``buffer.combine`` home. ``local_input_permutation_mapping`` carries the
    cache_id from :func:`token_pre_all2all`."""
    cache_id = int(local_input_permutation_mapping.item())
    state = _state_cache.pop(cache_id)

    if state.sorted_scores.numel() > 0:
        weighted = expert_outputs * state.sorted_scores.to(expert_outputs.dtype).reshape(-1, 1)
    else:
        weighted = expert_outputs

    unsorted = _unpermute_by_expert(weighted, state.sort_indices, state.num_recv)
    combined = _Combine.apply(unsorted, state.handle, ep_group)
    return combined.view(org_hidden_states_shape).to(state.input_dtype)
