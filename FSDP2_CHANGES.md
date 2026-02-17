# FSDP2 Integration for Nanotron

This document summarizes all changes made to add FSDP2 (PyTorch's composable `fully_shard` API) support to the Nanotron training framework.

## Overview

FSDP2 enables full parameter, gradient, and optimizer state sharding across data-parallel ranks (ZeRO-3 equivalent). Unlike Nanotron's existing ZeRO-1 (`ZeroDistributedOptimizer`), FSDP2 shards everything — not just optimizer states — reducing per-GPU memory proportionally to the DP world size.

The implementation supports:
- **Full sharding**: shard across all DP ranks (1D mesh)
- **Hybrid sharding (HSDP)**: shard within node, replicate across nodes (2D mesh)
- **`reshard_after_forward`**: configurable — `true` (ZeRO-3) or `false` (ZeRO-2-like)
- **Composition with Tensor Parallelism (TP)**: builds 2D DeviceMesh `(dp, tp)` for correct collective operations

## Validation

### DP-only (TP=1): 4x H100, bf16, 10 steps

| Comparison              | Max Loss Rel Diff | Max Grad Norm Rel Diff |
|-------------------------|-------------------|------------------------|
| DDP vs FSDP2            | 0.0014%           | 0.0108%                |
| DDP vs FSDP2_HYBRID     | 0.0015%           | 0.0138%                |
| DDP vs FSDP2_NORESHARD  | 0.0016%           | 0.0221%                |

### TP+FSDP2 (TP=2, DP=2): 4x H100, bf16, 10 steps

| Comparison                      | Max Loss Rel Diff | Max Grad Norm Rel Diff |
|---------------------------------|-------------------|------------------------|
| DDP_TP2 vs FSDP2_TP2            | 0.0033%           | 0.0197%                |
| DDP_TP2 vs FSDP2_TP2_NORESHARD  | 0.0022%           | 0.0195%                |

Run validation:
```bash
python validate_fsdp2_nanotron.py --steps 10 --nproc 4              # DP-only
python validate_fsdp2_nanotron.py --steps 10 --nproc 4 --tp 2      # TP + FSDP2
```

---

## New Files

### `src/nanotron/parallel/data_parallel/fsdp.py`
Core FSDP2 integration module. Key functions:
- `_build_dp_mesh()` — builds a `DeviceMesh` for FSDP2. When TP > 1, constructs a 2D mesh `(dp, tp)` covering all world ranks (required since `DeviceMesh` creation is collective) and extracts the `"dp"` sub-mesh. When TP=1, uses a simple 1D mesh.
- `apply_fsdp2()` — walks the model's `PipelineBlock`s, applies `fully_shard()` to each `pp_block` (skipping non-Module blocks like `cast_to_fp32`) and the root model.
- `collect_tied_param_metadata()` — captures NanotronParameter tied/sharded metadata *before* FSDP2 converts them to DTensors. This is critical for TP+FSDP2 to correctly handle gradient norm computation and tied gradient sync.
- `sync_fsdp_tied_gradients()` — manually syncs tied param gradients for FSDP2 models (needed for TP REDUCE_SCATTER mode where replicated param gradients must be summed across TP ranks).
- `is_fsdp_model()` — detects FSDP2-wrapped models via `_nanotron_fsdp` marker or PyTorch's internal `_fsdp_state`.
- `fsdp_no_sync()` / `fsdp_enable_sync()` — context managers for gradient accumulation using `set_requires_gradient_sync`.

### `validate_fsdp2_nanotron.py`
Validation script that runs DDP and FSDP2 training sequentially with identical configs, parses per-step losses and gradient norms from logs, and compares them. Uses high-precision logging mode for 8+ significant figures.

---

## Modified Files

### Configuration

#### `src/nanotron/config/parallelism_config.py`
- Added `DataParallelEngine` enum with values `DDP` and `FSDP2`.
- Added fields to `ParallelismArgs`:
  - `dp_engine: Optional[DataParallelEngine]` — selects DDP or FSDP2 (defaults to DDP)
  - `fsdp_reshard_after_forward: bool` — whether to free params after forward (default `True`)
  - `fsdp_hybrid: bool` — enable HSDP (default `False`)
- Added `is_fsdp` property for convenient checking.

#### `src/nanotron/config/config.py`
- Added `DataParallelEngine` to dacite `type_hooks` for YAML deserialization.

#### `src/nanotron/config/utils_config.py`
- Added generic `Enum` handling in `serialize()` so `DataParallelEngine` values serialize correctly to YAML.

### Training Loop

#### `src/nanotron/trainer.py`
**Key architectural change**: FSDP2 wrapping is deferred until *after* weight initialization.

- `_init_model()` — FSDP2 section now only runs `check_model_has_grad()` (no wrapping). DDP path unchanged.
- `_apply_data_parallelism()` — **new method** called from `init_model()` after `_load_model_checkpoint()`. First collects tied parameter metadata via `collect_tied_param_metadata()`, then applies `apply_fsdp2()`. This ordering ensures NanotronParameter attributes are captured before FSDP2 converts them to DTensors.
- `training_step()` — for FSDP2: calls `sync_fsdp_tied_gradients()` using pre-captured metadata (replaces `sync_tied_weights_gradients()` which can't work with DTensors). Passes `fsdp_tied_metadata` to `clip_grad_norm()` for correct tied param deduplication.
- `save_checkpoint()` — for FSDP models: all ranks participate in weight save (collective `dcp.save`), skips optimizer/LR-scheduler save (standard optimizer, not ZeRO).
- Gradient clipping call passes `dp_pg` when FSDP is active (needed since each rank holds only its shard's gradients).

#### `src/nanotron/parallel/pipeline_parallel/engine.py`
- `_get_bwd_context()` in `AllForwardAllBackwardPipelineEngine` — uses `fsdp_no_sync` for non-final microbatches and `fsdp_enable_sync` for the last microbatch when FSDP is active. This replaces DDP's `no_sync()` for gradient accumulation.

### Optimizer & Gradients

#### `src/nanotron/helpers.py`
- `init_optimizer_and_grad_accumulator()` — when FSDP is active:
  - Skips `ZeroDistributedOptimizer` (FSDP2 handles state sharding)
  - Disables `accumulate_grad_in_fp32` (incompatible with FSDP2's DTensors)
  - Sets `torch_adam_is_fused=False` (DTensors may not be contiguous)
- `get_custom_weight_decay_for_named_parameters()` — uses `getattr(param, "is_tied", False)` for DTensor safety.

#### `src/nanotron/optim/clip_grads.py`
- Added `_to_local_tensor()` helper — converts DTensors to their local shard tensor, returns regular tensors as-is.
- `clip_grad_norm()` now accepts optional `dp_pg` and `fsdp_tied_metadata` parameters. When `dp_pg` is provided, adds an all-reduce across DP ranks before the existing MP all-reduce. When `fsdp_tied_metadata` is provided, uses it to exclude tied params from non-primary ranks (replacing NanotronParameter-based `is_tied` checks that don't work with DTensors). This prevents over-counting replicated param gradients across TP ranks.
- All gradient access converted through `_to_local_tensor()` to avoid DTensor dispatch errors with raw process groups.

### DTensor Compatibility

FSDP2 converts `NanotronParameter` instances to `DTensor`s, which don't have custom attributes like `is_tied` or methods like `get_tied_info()`. The following files were updated to use `getattr(param, "is_tied", False)` instead of direct attribute access:

#### `src/nanotron/models/base.py`
- `get_named_params_with_correct_tied()` — safe access to `is_tied`.

#### `src/nanotron/parallel/tied_parameters.py`
- `get_tied_id_to_param()` — safe access to `is_tied`.

#### `src/nanotron/sanity_checks.py`
- `after_tbi_sanity_checks()` and `before_optim_step_sanity_checks()` — safe access to `is_tied`.

### Checkpointing

#### `src/nanotron/serialize/weights.py`
- Added `_is_fsdp_model()` detection helper.
- Added `_save_weights_fsdp()` — uses `torch.distributed.checkpoint.save()` with `get_model_state_dict()` which handles DTensors natively. Saves sharded checkpoints (`.distcp` format).
- `save_weights()` dispatches to the FSDP path when the model has `_nanotron_fsdp` marker.

#### `src/nanotron/serialize/main.py`
- `save()` — skips `NanotronParameter`-based sanity checks (tied param sync, optimizer state sync) for FSDP models, since parameters are DTensors.

### Logging

#### `src/nanotron/logging/base.py`
- `human_format()` — added high-precision mode (`NANOTRON_LOG_PRECISION=high` env var) that outputs 8 significant figures instead of 3, used by the validation script.

### Benchmark Scripts

#### `run_nanotron_benchmark.sh`
- Added four FSDP2 strategies: `fsdp2_full_reshard`, `fsdp2_full_noreshard`, `fsdp2_hybrid_reshard`, `fsdp2_hybrid_noreshard`.
- Config generation populates `dp_engine`, `fsdp_reshard_after_forward`, and `fsdp_hybrid` based on strategy.
- Sets `accumulate_grad_in_fp32: false` for FSDP2 strategies.

#### `submit_nanotron_benchmarks.sh`
- Added `--fsdp` experiment mode that sets `STRATEGIES` to all four FSDP2 variants.

---

## Configuration Example

```yaml
# FSDP2 only (TP=1)
parallelism:
  dp: 8
  pp: 1
  tp: 1
  dp_engine: fsdp2                    # "ddp" (default) or "fsdp2"
  fsdp_reshard_after_forward: true    # true = ZeRO-3, false = ZeRO-2-like
  fsdp_hybrid: false                  # true = HSDP (shard within node)

# TP + FSDP2 (e.g., 16 GPUs: TP=2, DP=8)
parallelism:
  dp: 8
  pp: 1
  tp: 2
  dp_engine: fsdp2
  fsdp_reshard_after_forward: true
```

## Design Decisions

1. **Deferred wrapping**: FSDP2 is applied *after* `init_model_randomly()` and checkpoint loading, because `fully_shard()` converts `NanotronParameter` to `DTensor`, losing custom attributes needed during initialization.

2. **Pre-captured metadata**: Before FSDP2 wrapping, we capture tied parameter metadata (`FSDPTiedParamMetadata`) — which params are tied, their global ranks, and reduce ops. This is essential for correct gradient norm computation (avoid double-counting replicated params across TP) and tied gradient synchronization (REDUCE_SCATTER mode).

3. **2D DeviceMesh for TP+FSDP**: When TP > 1, `DeviceMesh` creation is collective (all world ranks must participate). We create a 2D mesh `(dp, tp)` that all ranks agree on, then extract the `"dp"` sub-mesh for `fully_shard()`. This correctly isolates FSDP sharding to the DP dimension.

4. **No ZeRO optimizer**: FSDP2 handles optimizer state sharding internally, so `ZeroDistributedOptimizer` is bypassed.

5. **No FP32 gradient accumulation**: FSDP2's reduce-scatter is incompatible with nanotron's FP32 gradient accumulation hooks. Instead, FSDP2's `MixedPrecisionPolicy` with `reduce_dtype=float32` provides numerically stable gradient reduction.

6. **Distributed checkpoint**: FSDP2 models use `torch.distributed.checkpoint` (`.distcp` format) instead of nanotron's per-parameter safetensors, because DTensors require collective save operations.

7. **Pipeline compatibility**: FSDP2 is applied per-`PipelineBlock.pp_block`, respecting the existing PP structure. Non-Module blocks (e.g., `cast_to_fp32` lambda) are skipped.

## Current Limitations

- **TP + HSDP**: Hybrid sharding combined with TP > 1 is not yet supported (would need a 3D mesh).
- **PP + TP + FSDP2**: Pipeline parallelism combined with TP + FSDP2 is not yet supported (each PP stage would need its own mesh coordination).
- **CP + FSDP2**: Context parallelism combined with FSDP2 is not yet supported.
