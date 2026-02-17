"""FSDP2 (fully_shard) integration for Nanotron.

This module provides utilities to apply PyTorch's FSDP2 (composable fully_shard API)
to Nanotron models, enabling full parameter/gradient/optimizer-state sharding across
data-parallel ranks (equivalent to ZeRO-3).

Supports:
  - Full sharding (shard across all DP ranks)
  - Hybrid sharding / HSDP (shard within node, replicate across nodes)
  - reshard_after_forward=True (ZeRO-3: reshard params after fwd, re-gather in bwd)
  - reshard_after_forward=False (ZeRO-2-like: keep gathered params through fwd+bwd)
  - Composition with Tensor Parallelism (TP)
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from nanotron import distributed as dist
from nanotron import logging
from nanotron.logging import log_rank
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock

logger = logging.get_logger(__name__)

# Lazy import to avoid hard dependency on torch >= 2.4
_fully_shard = None
_MixedPrecisionPolicy = None


@dataclass
class FSDPTiedParamMetadata:
    """Preserves NanotronParameter tied/sharded metadata before FSDP2 converts them to DTensors.

    FSDP2's `fully_shard` converts NanotronParameters to DTensors in-place,
    losing the `is_tied`, `get_tied_info()`, and `is_sharded` metadata.
    This metadata is needed for:
    - Correct gradient norm computation (avoid double-counting replicated params across TP)
    - Tied gradient synchronization across TP ranks (for REDUCE_SCATTER mode)
    """
    # Param names that should be EXCLUDED from gradient norm on this rank
    # (tied params where this rank is not the primary rank)
    excluded_from_norm: Set[str] = field(default_factory=set)

    # Param names -> (group_ranks, reduce_op) that need gradient sync after backward
    # (tied params with reduce_op != None, e.g., replicated params in REDUCE_SCATTER mode)
    needs_grad_sync: Dict[str, Tuple[Tuple[int, ...], dist.ReduceOp]] = field(default_factory=dict)


def collect_tied_param_metadata(
    model: nn.Module,
    parallel_context: ParallelContext,
) -> FSDPTiedParamMetadata:
    """Collect tied parameter metadata before FSDP2 wrapping.

    Must be called BEFORE apply_fsdp2() since FSDP2 converts NanotronParameters to DTensors.
    The returned metadata is stored on the model and used during training for:
    - Gradient norm computation: skip tied params on non-primary ranks
    - Gradient sync: manually sync tied param gradients across TP ranks
    """
    metadata = FSDPTiedParamMetadata()
    world_rank = dist.get_rank()

    # Build module_id -> prefix mapping for resolving tied param names
    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
    module_id_to_prefix[id(model)] = ""

    for name, param in model.named_parameters():
        if not isinstance(param, NanotronParameter):
            continue

        if param.is_tied:
            tied_info = param.get_tied_info()
            full_name = tied_info.get_full_name_from_module_id_to_prefix(
                module_id_to_prefix=module_id_to_prefix
            )

            # For gradient norm: only the primary rank (global_ranks[0]) should count this param
            if world_rank != tied_info.global_ranks[0]:
                metadata.excluded_from_norm.add(full_name)

            # For gradient sync: if reduce_op is not None, this param needs explicit sync
            if tied_info.reduce_op is not None:
                group_ranks = tied_info.global_ranks
                metadata.needs_grad_sync[full_name] = (group_ranks, tied_info.reduce_op)

    log_rank(
        f"[FSDP2] Collected tied param metadata: "
        f"{len(metadata.excluded_from_norm)} excluded from norm, "
        f"{len(metadata.needs_grad_sync)} need grad sync",
        logger=logger, level=logging.INFO, rank=0,
    )

    return metadata


def _ensure_fsdp2_imports():
    global _fully_shard, _MixedPrecisionPolicy
    if _fully_shard is not None:
        return
    try:
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

        _fully_shard = fully_shard
        _MixedPrecisionPolicy = MixedPrecisionPolicy
    except ImportError as e:
        raise ImportError(
            "FSDP2 requires PyTorch >= 2.4. Please upgrade your PyTorch installation."
        ) from e


def _build_dp_mesh(
    parallel_context: ParallelContext,
    hybrid: bool = False,
) -> DeviceMesh:
    """Build a DeviceMesh for the DP group, compatible with TP > 1.

    When TP > 1, DeviceMesh creation is collective across all world ranks.
    We must construct a global mesh that all ranks agree on, then extract
    the DP sub-mesh for FSDP2.

    Args:
        parallel_context: Nanotron's parallel context.
        hybrid: If True, build an HSDP mesh (shard within node, replicate across nodes).

    Returns:
        A DeviceMesh suitable for passing to fully_shard().
    """
    import numpy as np

    dp_size = parallel_context.data_parallel_size
    tp_size = parallel_context.tensor_parallel_size
    pp_size = parallel_context.pipeline_parallel_size
    cp_size = parallel_context.context_parallel_size
    ep_size = parallel_context.expert_parallel_size
    world_size = parallel_context.world_size

    if tp_size > 1 and (pp_size > 1 or cp_size > 1 or ep_size > 1):
        raise NotImplementedError(
            f"FSDP2 + TP with PP={pp_size}, CP={cp_size}, or EP={ep_size} > 1 "
            f"is not yet supported. Use PP=1, CP=1, EP=1 with TP + FSDP2."
        )

    if tp_size > 1 and hybrid:
        # TP + HSDP: Build a global 3D mesh (replicate, shard, tp).
        # Nanotron rank layout: rank = dp_idx * tp_size + tp_idx
        # where dp_idx = replica_idx * shard_size + shard_idx
        #
        # HSDP shards within a "local group" (intra-node) and replicates across (inter-node).
        # With TP, there are (local_world_size / tp_size) DP positions per node.
        import os
        local_size = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
        shard_size = local_size // tp_size  # DP ranks per node

        if shard_size < 1:
            raise ValueError(
                f"TP + HSDP requires LOCAL_WORLD_SIZE ({local_size}) >= tp_size ({tp_size})."
            )
        if dp_size % shard_size != 0:
            log_rank(
                f"[FSDP2] WARNING: dp_size={dp_size} not divisible by shard_size={shard_size} "
                f"(local_size={local_size}, tp={tp_size}). Falling back to TP + full FSDP.",
                logger=logger, level=logging.WARNING, rank=0,
            )
            mesh_tensor = np.arange(world_size).reshape(dp_size, tp_size).tolist()
            mesh_2d = DeviceMesh("cuda", mesh_tensor, mesh_dim_names=("dp", "tp"))
            return mesh_2d["dp"]

        num_replicas = dp_size // shard_size

        if num_replicas <= 1:
            log_rank(
                f"[FSDP2] WARNING: Only {num_replicas} replica(s) for HSDP — degenerates to full sharding. "
                f"(dp={dp_size}, shard_size={shard_size}). Falling back to TP + full FSDP.",
                logger=logger, level=logging.WARNING, rank=0,
            )
            mesh_tensor = np.arange(world_size).reshape(dp_size, tp_size).tolist()
            mesh_2d = DeviceMesh("cuda", mesh_tensor, mesh_dim_names=("dp", "tp"))
            return mesh_2d["dp"]

        # 3D mesh: (num_replicas, shard_size, tp_size)
        # mesh[rep][shard][tp] = (rep * shard_size + shard) * tp_size + tp
        mesh_tensor = np.arange(world_size).reshape(num_replicas, shard_size, tp_size).tolist()
        mesh_3d = DeviceMesh(
            "cuda", mesh_tensor,
            mesh_dim_names=("replicate", "shard", "tp"),
        )
        hsdp_mesh = mesh_3d["replicate", "shard"]

        log_rank(
            f"[FSDP2] Built 3D (replicate, shard, tp) mesh: "
            f"replicas={num_replicas}, shard={shard_size}, tp={tp_size}, "
            f"mesh={mesh_tensor} -> using ('replicate', 'shard') sub-mesh for HSDP+TP",
            logger=logger, level=logging.INFO, rank=0,
        )
        return hsdp_mesh

    if tp_size > 1:
        # TP > 1 (non-hybrid): Build a global 2D mesh (dp, tp) that ALL ranks agree on.
        # DeviceMesh creation is collective (uses dist.new_group internally),
        # so all ranks must provide the same mesh tensor.
        #
        # Nanotron rank layout: (ep, pp, dp, cp, tp) = (1, 1, dp, 1, tp)
        # So ranks are ordered as: rank = dp_idx * tp_size + tp_idx
        mesh_tensor = np.arange(world_size).reshape(dp_size, tp_size).tolist()
        mesh_2d = DeviceMesh("cuda", mesh_tensor, mesh_dim_names=("dp", "tp"))
        dp_mesh = mesh_2d["dp"]

        log_rank(
            f"[FSDP2] Built 2D (dp, tp) mesh: dp={dp_size}, tp={tp_size}, "
            f"mesh={mesh_tensor} -> using 'dp' sub-mesh for FSDP",
            logger=logger, level=logging.INFO, rank=0,
        )
        return dp_mesh

    # TP == 1: All ranks share the same dp_ranks, so 1D mesh is safe
    dp_ranks = list(dist.get_process_group_ranks(parallel_context.dp_pg))

    if not hybrid or dp_size <= 1:
        mesh = DeviceMesh("cuda", dp_ranks)
        log_rank(
            f"[FSDP2] Built 1D DP mesh: ranks={dp_ranks}",
            logger=logger, level=logging.INFO, rank=0,
        )
        return mesh

    # HSDP (TP=1 only): shard within node, replicate across nodes
    import os
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))

    if dp_size % local_size != 0:
        log_rank(
            f"[FSDP2] WARNING: dp_size={dp_size} not divisible by local_size={local_size}. "
            f"Falling back to full (non-hybrid) sharding.",
            logger=logger, level=logging.WARNING, rank=0,
        )
        return DeviceMesh("cuda", dp_ranks)

    num_replicas = dp_size // local_size
    mesh_2d = [dp_ranks[i * local_size : (i + 1) * local_size] for i in range(num_replicas)]
    mesh = DeviceMesh("cuda", mesh_2d, mesh_dim_names=("replicate", "shard"))
    log_rank(
        f"[FSDP2] Built 2D HSDP mesh: {num_replicas} replicas x {local_size} shards, ranks={mesh_2d}",
        logger=logger, level=logging.INFO, rank=0,
    )
    return mesh


def apply_fsdp2(
    model: nn.Module,
    parallel_context: ParallelContext,
    reshard_after_forward: bool = True,
    hybrid: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Apply FSDP2 (fully_shard) to a Nanotron model.

    Wraps each transformer decoder layer's pp_block and the root model
    with fully_shard(). This enables automatic parameter sharding,
    gradient reduce-scatter, and optimizer state sharding.

    Args:
        model: The NanotronModel (not DDP-wrapped).
        parallel_context: Nanotron's parallel context.
        reshard_after_forward: If True (default), reshard parameters after
            forward pass. Saves memory but adds communication in backward.
        hybrid: If True, use hybrid sharding (HSDP).
        dtype: Model dtype for mixed precision policy.

    Returns:
        The same model, modified in-place with FSDP2.
    """
    _ensure_fsdp2_imports()

    mesh = _build_dp_mesh(parallel_context, hybrid=hybrid)

    # Mixed precision: keep params in bf16/fp16, compute in same dtype,
    # reduce gradients in fp32 for numerical stability
    mp_policy = _MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=torch.float32,
    )

    current_pp_rank = dist.get_rank(parallel_context.pp_pg)
    fsdp_wrapped_count = 0

    # Walk through all PipelineBlocks and wrap their pp_block (the actual compute module)
    for name, module in model.named_modules():
        if not isinstance(module, PipelineBlock):
            continue
        # Only wrap blocks assigned to the current PP rank
        if not hasattr(module, "rank") or module.rank != current_pp_rank:
            continue
        if not hasattr(module, "pp_block"):
            continue

        pp_block = module.pp_block
        # pp_block can be a plain function (not nn.Module), e.g. cast_to_fp32 — skip those
        if not isinstance(pp_block, nn.Module):
            continue
        # Only wrap modules that have parameters
        if sum(1 for _ in pp_block.parameters()) == 0:
            continue

        _fully_shard(
            pp_block,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )
        fsdp_wrapped_count += 1
        log_rank(
            f"[FSDP2] Wrapped {name}.pp_block ({pp_block.__class__.__name__})",
            logger=logger, level=logging.DEBUG, rank=0,
        )

    # Apply FSDP to the root model (handles any remaining un-wrapped parameters)
    _fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        mp_policy=mp_policy,
    )
    fsdp_wrapped_count += 1

    # Mark model so is_fsdp_model() can detect it reliably
    model._nanotron_fsdp = True
    # Store the DP process group for tied gradient sync
    model._nanotron_fsdp_dp_pg = parallel_context.dp_pg

    log_rank(
        f"[FSDP2] Applied fully_shard to {fsdp_wrapped_count} modules "
        f"(reshard_after_forward={reshard_after_forward}, hybrid={hybrid})",
        logger=logger, level=logging.INFO, rank=0,
    )

    return model


def is_fsdp_model(model: nn.Module) -> bool:
    """Check if a model has been wrapped with FSDP2 (fully_shard).

    FSDP2 doesn't wrap in a new class — it modifies the module in-place
    and registers hooks. We detect it by checking for FSDP2's internal
    state markers on the module or its children.
    """
    # Check the _is_fsdp marker we set ourselves (most reliable)
    if getattr(model, "_nanotron_fsdp", False):
        return True

    # Fallback: check for PyTorch FSDP2's internal state attributes
    for module in model.modules():
        if hasattr(module, "_fsdp_state") and module._fsdp_state is not None:
            return True
    return False


def sync_fsdp_tied_gradients(
    model: nn.Module,
    metadata: FSDPTiedParamMetadata,
    parallel_context: ParallelContext,
):
    """Sync tied parameter gradients for FSDP2 models.

    With FSDP2, NanotronParameter metadata is lost (DTensors). This function
    uses the pre-computed metadata to perform the gradient sync that
    `sync_tied_weights_gradients` would normally do.

    This is needed for:
    - TP REDUCE_SCATTER mode: unsharded params (LayerNorm, etc.) need gradient SUM across TP
    - PP > 1: embeddings-lm_head tied params need gradient sync across PP ranks
    """
    if not metadata.needs_grad_sync:
        return

    # Build name -> param mapping
    name_to_param = dict(model.named_parameters())

    # Group by (group_ranks, reduce_op) for coalesced all-reduce
    from collections import OrderedDict
    group_to_grads = OrderedDict()

    for param_name, (group_ranks, reduce_op) in sorted(metadata.needs_grad_sync.items()):
        param = name_to_param.get(param_name)
        if param is None or param.grad is None:
            continue

        grad = param.grad
        # Convert DTensor grad to local tensor for all-reduce
        if hasattr(grad, "to_local"):
            grad = grad.to_local()

        key = (group_ranks, reduce_op)
        if key not in group_to_grads:
            group_to_grads[key] = []
        group_to_grads[key].append(grad)

    for (group_ranks, reduce_op), grads in group_to_grads.items():
        pg = parallel_context.world_ranks_to_pg.get(group_ranks)
        if pg is None:
            continue
        dist.all_reduce_coalesced(tensors=grads, op=reduce_op, group=pg)


@contextmanager
def fsdp_no_sync(model: nn.Module):
    """Context manager to disable gradient sync in FSDP2.

    With FSDP2, gradient sync is controlled via set_requires_gradient_sync.
    This is used during gradient accumulation: skip sync on non-last microbatches.
    """
    try:
        from torch.distributed._composable.fsdp import set_requires_gradient_sync
    except ImportError:
        yield
        return

    old_state = True
    try:
        set_requires_gradient_sync(model, requires_gradient_sync=False)
        yield
    finally:
        set_requires_gradient_sync(model, requires_gradient_sync=old_state)


@contextmanager
def fsdp_enable_sync(model: nn.Module):
    """Context manager to explicitly enable gradient sync in FSDP2.

    Used on the last microbatch to ensure gradients are synchronized.
    """
    try:
        from torch.distributed._composable.fsdp import set_requires_gradient_sync
    except ImportError:
        yield
        return

    try:
        set_requires_gradient_sync(model, requires_gradient_sync=True)
        yield
    finally:
        pass
