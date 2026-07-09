# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Expert-parallel token communication for the weight-space MoE.

NCCL ``all_to_all_single`` dispatch/combine, ported from gen-ar's
``hy_parallelism.models.modules.moe.moe_parallel`` (+ ``moe_utils``) but stripped of the
grouped-GEMM / JVP / fused-kernel deps. Expert compute stays as our per-expert ``nn.Module``s
(a Python loop over the local experts), so this file only owns the *communication*:

    gate -> expert_mask
      preprocess            : all_gather per-expert token counts -> all-to-all split sizes
      token_pre_all2all     : permute local tokens, all-to-all to the expert owner, regroup by
                              local expert  ->  tokens grouped [local_expert_0 | local_expert_1 | ...]
      (caller runs the local experts on each group)
      tokens_post_all2all   : regroup back, all-to-all home, unpermute + top-k weight + sum

The result is **bit-for-bit** the dense top-k masked combine (validated by the CPU unit test),
so ``ep_size == 1`` is a no-op identity path.

Backend selector: ``nccl`` (this file) now; ``deepep`` later (same 3-function signature) once a
torch-2.12 / py3.10 build of deep_ep is available.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.distributed as dist

from ...utils.ep import get_ep_backend


def _tolist(x: Union[torch.Tensor, Sequence[int]]) -> list:
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return list(x)


# ---------------------------------------------------------------------------
# permutation helpers (ported from moe_utils.py)
# ---------------------------------------------------------------------------
def permute(tokens: torch.Tensor, routing_map: torch.Tensor):
    """Permute ``tokens`` [num_tokens, hidden] into expert-contiguous order given the sparse
    ``routing_map`` [num_experts, num_tokens] (0/1). Returns the permuted tokens and the
    row index map used to scatter them home in :func:`unpermute`."""
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map)
    permuted_input = tokens.index_select(0, sorted_indices)
    return permuted_input, sorted_indices


def unpermute(
    tokens: torch.Tensor,
    hidden_states_shape: torch.Size,
    permutation_mapping: torch.Tensor,
    routing_map: torch.Tensor,
    routing_weights: Optional[torch.Tensor] = None,
):
    """Inverse of :func:`permute`; scatter-add the (optionally top-k-weighted) expert outputs
    back to their original token rows. ``routing_weights`` is the dense [num_tokens, num_experts]
    gate produced by :func:`generate_weights_idx`."""
    if routing_weights is not None:
        tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())
        tokens = tokens * tokens_weight.unsqueeze(-1)

    hidden_dim = hidden_states_shape[-1]
    unpermuted_tokens = torch.zeros(hidden_states_shape, device=tokens.device, dtype=tokens.dtype)
    unpermuted_tokens.scatter_add_(
        0, permutation_mapping.unsqueeze(1).expand(-1, hidden_dim), tokens
    )
    return unpermuted_tokens


def generate_weights_idx(
    routing_weights: torch.Tensor, selected_experts: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Scatter the top-k ``routing_weights`` [T, k] into a dense [T, num_experts] gate at the
    ``selected_experts`` [T, k] columns (0 elsewhere)."""
    num_tokens, _ = routing_weights.shape
    weights_idx = torch.zeros(
        (num_tokens, num_experts), dtype=routing_weights.dtype, device=routing_weights.device
    )
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    return weights_idx


def sort_chunks_by_idxs(
    input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: Union[torch.Tensor, Sequence[int]]
) -> torch.Tensor:
    """Split ``input`` along dim-0 into chunks of ``split_sizes`` and concatenate them in
    ``sorted_idxs`` order (differentiable: ``split`` + ``cat``)."""
    chunks = torch.split(input, _tolist(split_sizes), dim=0)
    order = _tolist(sorted_idxs)
    return torch.cat([chunks[i] for i in order], dim=0)


# ---------------------------------------------------------------------------
# all-to-all (autograd-aware, NCCL / gloo)
# ---------------------------------------------------------------------------
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(group, input, output_split_sizes, input_split_sizes):
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return input
        input = input.contiguous()
        if output_split_sizes is None:
            output = torch.empty_like(input)
        else:
            output = torch.empty(
                size=(sum(output_split_sizes), input.size(1)),
                dtype=input.dtype,
                device=input.device,
            )
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        group, input, output_split_sizes, input_split_sizes = inputs
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

    @staticmethod
    def backward(ctx, *grad_output):
        # swap split sizes on the way back
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


def all_to_all(group, input, output_split_size=None, input_split_size=None):
    return _AllToAll.apply(group, input, output_split_size, input_split_size)


# ---------------------------------------------------------------------------
# dispatch / combine (ported from moe_parallel.py, group-gemm removed)
# ---------------------------------------------------------------------------
def _nccl_preprocess(expert_mask: torch.Tensor, num_experts: int, ep_group: dist.ProcessGroup):
    """Compute the all-to-all split sizes from the routing map.

    Args:
        expert_mask: [num_experts, top_k, num_tokens] one-hot dispatch mask.
        num_experts: global expert count.
        ep_group: the expert-parallel process group.

    Returns:
        input_splits  (list[int], len ep_size): #local tokens sent to each EP rank.
        output_splits (list[int], len ep_size): #tokens received from each EP rank.
        num_global_tokens_per_local_expert (LongTensor [ep_size, num_local_experts], CPU):
            per (source-rank, local-expert) received-token counts.
        num_global_sum_tokens_per_local_expert (LongTensor [num_local_experts], CPU):
            per-local-expert total received-token counts (used to slice the expert loop).
    """
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))  # [num_experts]

    input_splits = _tolist(
        num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1)
    )

    # all_gather the per-expert counts -> [ep_size, num_experts]. Use the list form (works on both
    # NCCL and gloo; the tensor is tiny, so the extra python objects are negligible).
    gathered = [torch.zeros_like(num_local_tokens_per_expert) for _ in range(ep_size)]
    dist.all_gather(gathered, num_local_tokens_per_expert.contiguous(), group=ep_group)
    num_global_tokens_per_expert = torch.stack(gathered, dim=0)

    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()

    output_splits = _tolist(num_global_tokens_per_local_expert.sum(dim=1))
    num_global_sum_tokens_per_local_expert = (
        num_global_tokens_per_local_expert.sum(dim=0).to(torch.device("cpu"))
    )
    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
        -1, num_local_experts
    ).to(torch.device("cpu"))

    return (
        input_splits,
        output_splits,
        num_global_tokens_per_local_expert,
        num_global_sum_tokens_per_local_expert,
    )


def _nccl_token_pre_all2all(
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
    input_splits,
    output_splits,
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_weights: Optional[torch.Tensor],  # unused here; NCCL applies the gate in tokens_post_all2all
    ep_group: dist.ProcessGroup,
):
    """Permute local tokens by expert, all-to-all to the owning EP rank, then regroup the
    received tokens so they are contiguous per local expert:
    ``[ local_expert_0's tokens | local_expert_1's tokens | ... ]``.

    Returns ``(global_permuted_hidden_states, routing_map, local_input_permutation_mapping,
    org_hidden_states_shape)`` — the last three are threaded into :func:`tokens_post_all2all`.
    """
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_hidden_states_shape = hidden_states.shape
    routing_map = expert_mask.sum(dim=1)  # [num_experts, num_tokens]

    local_permuted_hidden_states, local_input_permutation_mapping = permute(hidden_states, routing_map)
    global_permuted_hidden_states = all_to_all(
        ep_group, local_permuted_hidden_states, output_splits, input_splits
    )

    num_local_experts = num_experts // ep_group.size()
    # regroup received chunks from (source_rank, local_expert) order to (local_expert, source_rank)
    permute_order = _tolist(torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel())
    global_permuted_hidden_states = sort_chunks_by_idxs(
        global_permuted_hidden_states,
        num_global_tokens_per_local_expert.ravel(),
        permute_order,
    )
    return (
        global_permuted_hidden_states,
        routing_map,
        local_input_permutation_mapping,
        org_hidden_states_shape,
    )


def _nccl_tokens_post_all2all(
    expert_outputs: torch.Tensor,
    selected_experts: torch.Tensor,
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
    """Inverse of :func:`_nccl_token_pre_all2all`: regroup expert outputs back to (source_rank,
    local_expert) order, all-to-all home, then unpermute with the top-k gate weights and sum."""
    num_local_experts = num_experts // ep_group.size()
    unpermute_order = _tolist(torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel())
    expert_outputs = sort_chunks_by_idxs(
        expert_outputs,
        num_global_tokens_per_local_expert.T.ravel(),
        unpermute_order,
    )

    unpermute_outputs = all_to_all(ep_group, expert_outputs, input_splits, output_splits)

    weights_idx = None
    if routing_weights is not None:
        weights_idx = generate_weights_idx(routing_weights, selected_experts, num_experts)

    unpermute_outputs = unpermute(
        unpermute_outputs,
        org_hidden_states_shape,
        local_input_permutation_mapping,
        routing_map,
        routing_weights=weights_idx,
    )
    return unpermute_outputs


# ---------------------------------------------------------------------------
# public API: dispatch to the selected backend ('nccl' | 'deepep')
# ---------------------------------------------------------------------------
def preprocess(expert_mask, num_experts, ep_group):
    if get_ep_backend() == "deepep":
        from . import moe_ep_deepep as _d
        return _d.preprocess(expert_mask, num_experts, ep_group)
    return _nccl_preprocess(expert_mask, num_experts, ep_group)


def token_pre_all2all(
    hidden_states, expert_mask, num_experts, input_splits, output_splits,
    num_global_tokens_per_local_expert, routing_weights, ep_group,
):
    if get_ep_backend() == "deepep":
        from . import moe_ep_deepep as _d
        return _d.token_pre_all2all(
            hidden_states, expert_mask, num_experts, input_splits, output_splits,
            num_global_tokens_per_local_expert, routing_weights, ep_group,
        )
    return _nccl_token_pre_all2all(
        hidden_states, expert_mask, num_experts, input_splits, output_splits,
        num_global_tokens_per_local_expert, routing_weights, ep_group,
    )


def tokens_post_all2all(
    expert_outputs, selected_experts, num_experts, input_splits, output_splits,
    num_global_tokens_per_local_expert, routing_map, local_input_permutation_mapping,
    org_hidden_states_shape, routing_weights, ep_group,
):
    if get_ep_backend() == "deepep":
        from . import moe_ep_deepep as _d
        return _d.tokens_post_all2all(
            expert_outputs, selected_experts, num_experts, input_splits, output_splits,
            num_global_tokens_per_local_expert, routing_map, local_input_permutation_mapping,
            org_hidden_states_shape, routing_weights, ep_group,
        )
    return _nccl_tokens_post_all2all(
        expert_outputs, selected_experts, num_experts, input_splits, output_splits,
        num_global_tokens_per_local_expert, routing_map, local_input_permutation_mapping,
        org_hidden_states_shape, routing_weights, ep_group,
    )
