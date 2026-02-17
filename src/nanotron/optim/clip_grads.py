from typing import TYPE_CHECKING, Iterable, Optional, Set, Tuple

import torch

import nanotron.distributed as dist
from nanotron import logging
from nanotron.optim.gradient_accumulator import GradientAccumulator
from nanotron.parallel.parameters import NanotronParameter

if TYPE_CHECKING:
    from nanotron.parallel.data_parallel.fsdp import FSDPTiedParamMetadata

logger = logging.get_logger(__name__)


def _to_local_tensor(t: torch.Tensor) -> torch.Tensor:
    """Convert a DTensor to its local shard tensor, or return regular tensors as-is."""
    if hasattr(t, "to_local"):
        return t.to_local()
    return t


@torch.profiler.record_function("clip_grad_norm")
def clip_grad_norm(
    mp_pg: dist.ProcessGroup,
    named_parameters: Iterable[Tuple[str, NanotronParameter]],
    max_norm: float,
    grad_accumulator: Optional[GradientAccumulator],
    norm_type: float = 2.0,
    dp_pg: Optional[dist.ProcessGroup] = None,
    fsdp_tied_metadata: Optional["FSDPTiedParamMetadata"] = None,
) -> torch.Tensor:
    """Clips gradients. Adapted from torch.nn.utils.clip_grad_norm_.
    Norms are computed in fp32 precision to retain most accuracy.

    Args:
        mp_pg (dist.ProcessGroup): Process group for model parallel, ie all the ranks part of the same model replica (TP x PP)
        named_parameters (Iterable[(str, Parameter)]): an iterable of named Parameters that will have gradients normalized.
        grad_accumulator (GradientAccumulator): grad accumulator. If not None, in case of Zero1, we need to clip all fp32 grads
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
        dp_pg (dist.ProcessGroup, optional): DP process group. Required for FSDP where each rank holds
            only a shard of the gradients and we need to reduce across DP ranks.
        fsdp_tied_metadata (FSDPTiedParamMetadata, optional): Pre-computed tied param metadata for FSDP2.
            When set, uses this to determine which params to exclude from norm computation
            (replaces NanotronParameter.is_tied checks that don't work with DTensors).

    .. note:: In case parameters contains tied weights, we keep only a single copy of the gradient, but modify the
        gradient of all tied weights.
    """
    named_parameters = list(named_parameters)
    world_rank = dist.get_rank()

    # assert that all params require grad
    for _, p in named_parameters:
        assert p.requires_grad, "clip_grad_norm_ only supports Tensors that require grad"

    # Determine which params to exclude from norm computation.
    # With FSDP2, NanotronParameter metadata (is_tied) is lost, so use pre-computed metadata.
    excluded_from_norm: Set[str] = (
        fsdp_tied_metadata.excluded_from_norm if fsdp_tied_metadata is not None else set()
    )

    if grad_accumulator is None:
        if fsdp_tied_metadata is not None:
            # FSDP2 path: use pre-computed metadata for tied param deduplication
            grads = [
                _to_local_tensor(p.grad) for name, p in named_parameters
                if name not in excluded_from_norm
            ]
        else:
            # DDP / standard path: use NanotronParameter.is_tied
            grads = [
                _to_local_tensor(p.grad) for _, p in named_parameters
                if not getattr(p, "is_tied", False) or world_rank == p.get_tied_info().global_ranks[0]
            ]
    else:
        # In case of FP32 Grad Accum, We need to clip all fp32 grads
        if fsdp_tied_metadata is not None:
            grads = [
                grad_accumulator.get_grad_buffer(name)
                for name, p in named_parameters
                if name not in excluded_from_norm
            ]
        else:
            grads = [
                grad_accumulator.get_grad_buffer(name)
                for name, p in named_parameters
                if not getattr(p, "is_tied", False) or world_rank == p.get_tied_info().global_ranks[0]
            ]

    # Calculate gradient norm
    if norm_type == torch.inf:
        if len(grads) > 0:
            total_norm = torch.max(
                torch.stack([torch.linalg.vector_norm(g.detach(), ord=torch.inf, dtype=torch.float) for g in grads])
            )
        else:
            total_norm = torch.zeros([], dtype=torch.float, device=torch.device("cuda"))
        # With FSDP, also reduce across DP ranks (different shards)
        if dp_pg is not None:
            dist.all_reduce(total_norm, group=dp_pg, op=dist.ReduceOp.MAX)
        dist.all_reduce(total_norm, group=mp_pg, op=dist.ReduceOp.MAX)

    else:
        if len(grads) > 0:
            total_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g.detach(), ord=norm_type, dtype=torch.float) for g in grads]),
                ord=norm_type,
                dtype=torch.float,
            ).pow(norm_type)
        else:
            total_norm = torch.zeros([], dtype=torch.float, device=torch.device("cuda"))
        # With FSDP, also reduce across DP ranks (different shards hold different grad portions)
        if dp_pg is not None:
            dist.all_reduce(total_norm, group=dp_pg, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_norm, group=mp_pg, op=dist.ReduceOp.SUM)
        total_norm.pow_(1.0 / norm_type)

    # Scale gradients
    clip_coef = max_norm / (total_norm + 1.0e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    devices = set()
    for name, param in named_parameters:
        if grad_accumulator is None:
            g = _to_local_tensor(param.grad)
            devices.add(g.device)
        else:
            devices.add(grad_accumulator.get_grad_buffer(name).device)
    device_to_clip_coef_clamped = {device: clip_coef_clamped.to(device) for device in devices}

    for name, param in named_parameters:
        if grad_accumulator is None:
            g = _to_local_tensor(param.grad)
            g.detach().mul_(device_to_clip_coef_clamped[g.device])
        else:
            grad_accumulator.get_grad_buffer(name).detach().mul_(
                device_to_clip_coef_clamped[grad_accumulator.get_grad_buffer(name).device]
            )

    return total_norm
