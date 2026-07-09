# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Expert-parallel (EP) process groups for the weight-space MoE.

Topology (per the "intra-node EP + inter-node replicate" design): with ``ep_size`` experts
parallelized over ``ep_size`` consecutive ranks (an EP group), and the ``world_size / ep_size``
EP groups replicated as expert-data-parallel (EDP) replicas.

  world ranks (accelerate multi-node): rank = machine_rank * gpus_per_node + local_rank
  ep_size == gpus_per_node  ->  each EP group is exactly one node (intra-node all-to-all),
  and EDP groups run across nodes (inter-node grad all-reduce of the expert shards).

Example world=16, ep_size=8:
  EP groups : [0..7] (node0), [8..15] (node1)               <- experts sharded within a node
  EDP groups: [0,8],[1,9],...,[7,15]                          <- same expert-slot across nodes

Every rank calls ``dist.new_group`` for ALL groups in the same order (collective requirement).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from .logger_utils import setup_logger

logger = setup_logger(__name__)

# Module-level EP state (set once by init_ep_groups; read by the MoE layer).
_EP_STATE: dict = {
    "enabled": False,
    "ep_size": 1,
    "ep_rank": 0,
    "ep_group": None,
    "edp_size": 1,
    "edp_rank": 0,
    "edp_group": None,
}


def init_ep_groups(ep_size: int) -> dict:
    """Build the EP + expert-DP process groups. Idempotent. Requires an initialized default
    process group (accelerate launch has done this by the time the model/trainer is built).

    Returns the EP state dict.
    """
    if ep_size <= 1:
        _EP_STATE.update(enabled=False, ep_size=1, ep_rank=0, ep_group=None,
                         edp_size=1, edp_rank=0, edp_group=None)
        return _EP_STATE
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            f"init_ep_groups(ep_size={ep_size}) requires an initialized torch.distributed "
            "process group (launch via accelerate/torchrun)."
        )
    if _EP_STATE["enabled"] and _EP_STATE["ep_size"] == ep_size:
        return _EP_STATE  # already built

    world = dist.get_world_size()
    rank = dist.get_rank()
    if world % ep_size != 0:
        raise ValueError(
            f"world_size ({world}) must be divisible by ep_size ({ep_size}) for expert parallelism."
        )
    n_ep_groups = world // ep_size

    # EP groups: consecutive ep_size ranks (== one node when ep_size == gpus_per_node).
    ep_group = None
    ep_rank = 0
    for g in range(n_ep_groups):
        ranks = list(range(g * ep_size, (g + 1) * ep_size))
        pg = dist.new_group(ranks=ranks)
        if rank in ranks:
            ep_group = pg
            ep_rank = rank - g * ep_size

    # Expert-DP groups: same EP-slot across the n_ep_groups groups (inter-node replicas).
    edp_group = None
    edp_rank = 0
    for slot in range(ep_size):
        ranks = [g * ep_size + slot for g in range(n_ep_groups)]
        pg = dist.new_group(ranks=ranks)
        if rank in ranks:
            edp_group = pg
            edp_rank = ranks.index(rank)

    _EP_STATE.update(
        enabled=True, ep_size=ep_size, ep_rank=ep_rank, ep_group=ep_group,
        edp_size=n_ep_groups, edp_rank=edp_rank, edp_group=edp_group,
    )
    logger.info(
        f"[EP] initialized: world={world} ep_size={ep_size} (ep_rank={ep_rank}) "
        f"edp_size={n_ep_groups} (edp_rank={edp_rank})"
    )
    return _EP_STATE


def ep_enabled() -> bool:
    return bool(_EP_STATE["enabled"])


def get_ep_group() -> Optional[dist.ProcessGroup]:
    return _EP_STATE["ep_group"]


def get_ep_size() -> int:
    return int(_EP_STATE["ep_size"])


def get_ep_rank() -> int:
    return int(_EP_STATE["ep_rank"])


def get_edp_group() -> Optional[dist.ProcessGroup]:
    return _EP_STATE["edp_group"]


def get_edp_size() -> int:
    return int(_EP_STATE["edp_size"])


def local_expert_indices(num_experts: int) -> list[int]:
    """Global expert indices owned by THIS EP rank: [ep_rank*L, (ep_rank+1)*L)."""
    ep_size = get_ep_size()
    if ep_size <= 1:
        return list(range(num_experts))
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})."
        )
    L = num_experts // ep_size
    r = get_ep_rank()
    return list(range(r * L, (r + 1) * L))
