#!/usr/bin/env python3
"""
Validate Nanotron FSDP2 numerical correctness against DDP baseline.

Runs DDP and FSDP2 training with identical configs in FP32, then compares
per-step losses and gradient norms to ensure they produce equivalent results.

Usage (needs 2+ GPUs in a GPU session):
    cd /home/david.romero/projects/fsdp-bench/nanotron
    python validate_fsdp2_nanotron.py [--steps N] [--nproc N]

The script launches two sequential torchrun processes (DDP then FSDP2),
parses their logs, and prints a comparison table.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


# =============================================================================
# Config generation
# =============================================================================


def make_config_yaml(
    dp_engine: str = "ddp",
    fsdp_reshard: bool = True,
    fsdp_hybrid: bool = False,
    num_gpus: int = 2,
    tp: int = 1,
    train_steps: int = 20,
    seq_len: int = 128,
    micro_batch_size: int = 2,
    dtype: str = "float32",
    seed: int = 42,
    ckpt_path: str = "/tmp/validate_fsdp2_ckpt",
) -> str:
    """Generate a minimal nanotron config YAML for validation."""
    dp = num_gpus // tp
    num_kv_heads = max(4 // tp, 1) * tp  # ensure divisible by tp
    num_attn_heads = num_kv_heads  # keep same for simplicity
    return textwrap.dedent(f"""\
        general:
          project: validate_fsdp2
          run: {dp_engine}_tp{tp}_run
          seed: {seed}
          ignore_sanity_checks: true

        parallelism:
          dp: {dp}
          pp: 1
          tp: {tp}
          dp_engine: {dp_engine}
          fsdp_reshard_after_forward: {str(fsdp_reshard).lower()}
          fsdp_hybrid: {str(fsdp_hybrid).lower()}

        model:
          model_config:
            is_llama_config: true
            hidden_size: 256
            intermediate_size: 512
            num_hidden_layers: 2
            num_attention_heads: {num_attn_heads}
            num_key_value_heads: {num_kv_heads}
            vocab_size: 4096
            max_position_embeddings: {seq_len}
            rms_norm_eps: 1.0e-5
            rope_theta: 10000.0
          init_method:
            std: 0.02
          dtype: {dtype}

        tokenizer:
          tokenizer_name_or_path: gpt2

        checkpoints:
          checkpoints_path: {ckpt_path}/{dp_engine}_tp{tp}
          checkpoint_interval: 999999

        logging:
          log_level: info

        tokens:
          sequence_length: {seq_len}
          train_steps: {train_steps}
          micro_batch_size: {micro_batch_size}
          batch_accumulation_per_replica: 1

        optimizer:
          zero_stage: 0
          weight_decay: 0.01
          clip_grad: 1.0
          accumulate_grad_in_fp32: false
          optimizer_factory:
            name: adamW
            adam_eps: 1.0e-8
            adam_beta1: 0.9
            adam_beta2: 0.95
            torch_adam_is_fused: false
          learning_rate_scheduler:
            learning_rate: 3.0e-4
            lr_warmup_steps: 2
            lr_warmup_style: linear
            lr_decay_style: cosine
            min_decay_lr: 1.0e-5

        data_stages:
          - name: train
            start_training_step: 1
            data:
              dataset: null
              seed: {seed}
              num_loading_workers: 0
    """)


# =============================================================================
# Log parsing
# =============================================================================

# Example log line:
#   iteration: 1 / 5 | consumed_tokens: 512 | ... | grad_norm: 2.06 | lm_loss: 8.36 | lr: 0.00015 | ...
ITER_PATTERN = re.compile(
    r"iteration:\s*(\d+)\s*/\s*\d+.*?"
    r"grad_norm:\s*([\d.eE+\-]+).*?"
    r"lm_loss:\s*([\d.eE+\-]+)"
)


def parse_training_log(output: str) -> list[dict]:
    """Parse nanotron training output for iteration, loss, and grad_norm."""
    results = []
    for line in output.splitlines():
        m = ITER_PATTERN.search(line)
        if m:
            results.append({
                "step": int(m.group(1)),
                "grad_norm": float(m.group(2)),
                "lm_loss": float(m.group(3)),
            })
    return results


# =============================================================================
# Training runner
# =============================================================================


def run_training(
    config_yaml: str,
    nproc: int,
    label: str,
    run_dir: str,
    extra_env: dict[str, str] | None = None,
) -> list[dict]:
    """Write config and run torchrun, returning parsed per-step metrics."""
    config_path = os.path.join(run_dir, f"config_{label}.yaml")
    with open(config_path, "w") as f:
        f.write(config_yaml)

    script_dir = Path(__file__).resolve().parent
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        str(script_dir / "run_train.py"),
        "--config-file",
        config_path,
    ]

    print(f"\n{'=' * 60}")
    print(f"Running {label} ({nproc} GPUs)")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = "disabled"
    env["NANOTRON_LOG_PRECISION"] = "high"
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    if extra_env:
        env.update(extra_env)
        print(f"Extra env: {extra_env}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(script_dir),
    )

    full_output = result.stdout + "\n" + result.stderr

    # Save full output for debugging
    log_path = os.path.join(run_dir, f"output_{label}.txt")
    with open(log_path, "w") as f:
        f.write(full_output)
    print(f"Full output saved to {log_path}")

    if result.returncode != 0:
        print(f"\nERROR: {label} training failed (exit code {result.returncode})")
        # Print lines containing actual errors/tracebacks
        lines = full_output.strip().splitlines()
        error_lines = [l for l in lines if any(kw in l.lower() for kw in
                       ["error", "traceback", "exception", "assert", "rank0"])]
        if error_lines:
            print("--- Key error lines ---")
            for line in error_lines[:40]:
                print(f"  {line}")
        # Also print last 20 lines
        print("--- Last 20 lines ---")
        for line in lines[-20:]:
            print(f"  {line}")
        sys.exit(1)

    metrics = parse_training_log(full_output)
    print(f"Parsed {len(metrics)} training steps")

    if metrics:
        for m in metrics[:3]:
            print(f"  step {m['step']}: loss={m['lm_loss']:.6f}, grad_norm={m['grad_norm']:.6f}")
        if len(metrics) > 3:
            print(f"  ...")
            m = metrics[-1]
            print(f"  step {m['step']}: loss={m['lm_loss']:.6f}, grad_norm={m['grad_norm']:.6f}")

    return metrics


# =============================================================================
# Comparison
# =============================================================================


def compare_results(
    name1: str,
    name2: str,
    metrics1: list[dict],
    metrics2: list[dict],
) -> bool:
    """Compare per-step metrics and print a detailed comparison table."""
    n = min(len(metrics1), len(metrics2))
    if n == 0:
        print("\nERROR: No metrics to compare!")
        return False

    print(f"\n{'=' * 70}")
    print(f"Comparison: {name1} vs {name2}  ({n} steps)")
    print(f"{'=' * 70}")

    # Loss comparison
    print(f"\n--- Loss Comparison ---")
    print(f"{'Step':>5} | {name1:>12} | {name2:>12} | {'Abs Diff':>10} | {'Rel Diff':>10}")
    print("-" * 60)

    max_loss_abs = 0.0
    max_loss_rel = 0.0
    for i in range(n):
        l1 = metrics1[i]["lm_loss"]
        l2 = metrics2[i]["lm_loss"]
        abs_diff = abs(l1 - l2)
        rel_diff = abs_diff / max(abs(l1), 1e-8) * 100
        max_loss_abs = max(max_loss_abs, abs_diff)
        max_loss_rel = max(max_loss_rel, rel_diff)
        print(f"{i+1:>5} | {l1:>12.6f} | {l2:>12.6f} | {abs_diff:>10.6f} | {rel_diff:>9.4f}%")

    # Grad norm comparison
    print(f"\n--- Gradient Norm Comparison ---")
    print(f"{'Step':>5} | {name1:>12} | {name2:>12} | {'Abs Diff':>10} | {'Rel Diff':>10}")
    print("-" * 60)

    max_grad_abs = 0.0
    max_grad_rel = 0.0
    for i in range(n):
        g1 = metrics1[i]["grad_norm"]
        g2 = metrics2[i]["grad_norm"]
        abs_diff = abs(g1 - g2)
        rel_diff = abs_diff / max(abs(g1), 1e-8) * 100
        max_grad_abs = max(max_grad_abs, abs_diff)
        max_grad_rel = max(max_grad_rel, rel_diff)
        print(f"{i+1:>5} | {g1:>12.6f} | {g2:>12.6f} | {abs_diff:>10.6f} | {rel_diff:>9.4f}%")

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"Max absolute loss difference:      {max_loss_abs:.8f}")
    print(f"Max relative loss difference:      {max_loss_rel:.4f}%")
    print(f"Max absolute grad norm difference: {max_grad_abs:.8f}")
    print(f"Max relative grad norm difference: {max_grad_rel:.4f}%")

    # Check initial step
    initial_loss_match = abs(metrics1[0]["lm_loss"] - metrics2[0]["lm_loss"]) < 0.01
    initial_grad_match = (
        abs(metrics1[0]["grad_norm"] - metrics2[0]["grad_norm"])
        / max(metrics1[0]["grad_norm"], 1e-8) < 0.05
    )

    print(f"\nInitial step match:")
    loss_diff_0 = abs(metrics1[0]["lm_loss"] - metrics2[0]["lm_loss"])
    grad_diff_0 = abs(metrics1[0]["grad_norm"] - metrics2[0]["grad_norm"])
    grad_rel_0 = grad_diff_0 / max(metrics1[0]["grad_norm"], 1e-8) * 100
    print(f"  Loss: {'PASS' if initial_loss_match else 'FAIL'} (diff={loss_diff_0:.8f})")
    print(f"  Grad: {'PASS' if initial_grad_match else 'FAIL'} (rel_diff={grad_rel_0:.4f}%)")

    passed = initial_loss_match and initial_grad_match and max_loss_rel < 5.0
    if passed:
        print(f"\nPASS: {name1} and {name2} produce equivalent results!")
    else:
        print(f"\nFAIL: Significant differences detected between {name1} and {name2}!")

    return passed


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Validate Nanotron FSDP2 numerical correctness against DDP"
    )
    parser.add_argument("--steps", type=int, default=20, help="Training steps (default: 20)")
    parser.add_argument("--nproc", type=int, default=2, help="Number of GPUs (default: 2)")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length (default: 128)")
    parser.add_argument("--micro-batch-size", type=int, default=2, help="Micro batch size (default: 2)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"],
                        help="Model dtype (default: float32 for cleaner comparison)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--tp", type=int, default=0,
                        help="If > 0, only run TP+FSDP2 validation with this TP degree (requires nproc > tp)")
    parser.add_argument("--hsdp", action="store_true",
                        help="Also test HSDP (hybrid sharding) variants. With --tp, tests TP+HSDP.")
    parser.add_argument("--local-world-size", type=int, default=0,
                        help="Override LOCAL_WORLD_SIZE for HSDP testing (simulate multi-node)")
    args = parser.parse_args()

    # Use a persistent directory under nanotron/ so logs survive compute-node cleanup
    script_dir = Path(__file__).resolve().parent
    run_dir = str(script_dir / "validate_runs" / f"run_{os.getpid()}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Working directory: {run_dir}")

    # Common config kwargs
    common = dict(
        num_gpus=args.nproc,
        train_steps=args.steps,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        dtype=args.dtype,
        seed=args.seed,
        ckpt_path=os.path.join(run_dir, "ckpts"),
    )

    all_passed = True

    if args.tp > 0:
        # =====================================================================
        # TP + FSDP2 validation mode
        # =====================================================================
        tp = args.tp
        assert args.nproc % tp == 0, f"nproc={args.nproc} must be divisible by tp={tp}"
        assert args.nproc > tp, f"nproc={args.nproc} must be > tp={tp} to have DP > 1"

        # Baseline: DDP with same TP degree
        ddp_tp_yaml = make_config_yaml(dp_engine="ddp", tp=tp, **common)
        ddp_tp_metrics = run_training(ddp_tp_yaml, args.nproc, f"DDP_TP{tp}", run_dir)

        # FSDP2 + TP (full shard)
        fsdp_tp_yaml = make_config_yaml(dp_engine="fsdp2", fsdp_reshard=True, tp=tp, **common)
        fsdp_tp_metrics = run_training(fsdp_tp_yaml, args.nproc, f"FSDP2_TP{tp}", run_dir)

        all_passed &= compare_results(f"DDP_TP{tp}", f"FSDP2_TP{tp}", ddp_tp_metrics, fsdp_tp_metrics)

        # FSDP2 + TP (no reshard)
        fsdp_tp_nr_yaml = make_config_yaml(dp_engine="fsdp2", fsdp_reshard=False, tp=tp, **common)
        fsdp_tp_nr_metrics = run_training(fsdp_tp_nr_yaml, args.nproc, f"FSDP2_TP{tp}_NORESHARD", run_dir)

        all_passed &= compare_results(f"DDP_TP{tp}", f"FSDP2_TP{tp}_NORESHARD", ddp_tp_metrics, fsdp_tp_nr_metrics)

        # HSDP + TP (hybrid sharding with tensor parallelism)
        if args.hsdp:
            dp = args.nproc // tp
            # Determine LOCAL_WORLD_SIZE for HSDP
            local_ws = args.local_world_size if args.local_world_size > 0 else int(os.environ.get("LOCAL_WORLD_SIZE", str(args.nproc)))
            shard_size = local_ws // tp
            num_replicas = dp // shard_size if shard_size > 0 else 0

            if shard_size < 1 or dp % shard_size != 0 or num_replicas <= 1:
                print(f"\nWARNING: Skipping TP+HSDP test — need meaningful HSDP topology.")
                print(f"  dp={dp}, tp={tp}, local_world_size={local_ws}, shard_size={shard_size}, replicas={num_replicas}")
                print(f"  Need: shard_size >= 1, dp % shard_size == 0, replicas > 1")
                print(f"  Try more GPUs or a different --local-world-size.")
            else:
                hsdp_env = {"LOCAL_WORLD_SIZE": str(local_ws)}

                fsdp_tp_hsdp_yaml = make_config_yaml(
                    dp_engine="fsdp2", fsdp_reshard=True, fsdp_hybrid=True, tp=tp, **common,
                )
                fsdp_tp_hsdp_metrics = run_training(
                    fsdp_tp_hsdp_yaml, args.nproc, f"FSDP2_TP{tp}_HSDP", run_dir,
                    extra_env=hsdp_env,
                )
                all_passed &= compare_results(
                    f"DDP_TP{tp}", f"FSDP2_TP{tp}_HSDP", ddp_tp_metrics, fsdp_tp_hsdp_metrics,
                )

    else:
        # =====================================================================
        # Standard (TP=1) validation mode
        # =====================================================================

        # Run DDP baseline
        ddp_yaml = make_config_yaml(dp_engine="ddp", **common)
        ddp_metrics = run_training(ddp_yaml, args.nproc, "DDP", run_dir)

        # Run FSDP2 (full shard)
        fsdp_yaml = make_config_yaml(dp_engine="fsdp2", fsdp_reshard=True, **common)
        fsdp_metrics = run_training(fsdp_yaml, args.nproc, "FSDP2", run_dir)

        # Run FSDP2 hybrid (HSDP)
        fsdp_hybrid_yaml = make_config_yaml(
            dp_engine="fsdp2", fsdp_reshard=True, fsdp_hybrid=True, **common,
        )
        fsdp_hybrid_metrics = run_training(fsdp_hybrid_yaml, args.nproc, "FSDP2_HYBRID", run_dir)

        # Run FSDP2 no-reshard
        fsdp_noreshard_yaml = make_config_yaml(
            dp_engine="fsdp2", fsdp_reshard=False, **common,
        )
        fsdp_noreshard_metrics = run_training(fsdp_noreshard_yaml, args.nproc, "FSDP2_NORESHARD", run_dir)

        # Compare all variants against DDP
        all_passed &= compare_results("DDP", "FSDP2", ddp_metrics, fsdp_metrics)
        all_passed &= compare_results("DDP", "FSDP2_HYBRID", ddp_metrics, fsdp_hybrid_metrics)
        all_passed &= compare_results("DDP", "FSDP2_NORESHARD", ddp_metrics, fsdp_noreshard_metrics)

    print(f"\nFull logs available in: {run_dir}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
