#!/bin/bash
# Submit Nanotron benchmark jobs via gypsum's sbatch.sh
#
# Usage:
#   ./submit_nanotron_benchmarks.sh [--dry-run] [--smoke-test] [--models 1b,3b] [--nodes 1,2,4,8]
#   ./submit_nanotron_benchmarks.sh --tp    [--dry-run] [--nodes 1,2,4,8]  # TP=2 experiments (LLaMA 8B)
#   ./submit_nanotron_benchmarks.sh --pp    [--dry-run] [--nodes 1,2,4,8]  # PP=2 experiments (LLaMA 8B)
#   ./submit_nanotron_benchmarks.sh --smol  [--dry-run] [--nodes 1,2,4,8]  # SmolLM3 3B recipe
#   ./submit_nanotron_benchmarks.sh --fsdp  [--dry-run] [--nodes 1,2,4,8]  # FSDP2 experiments
#
# Strategies: ddp, zero1, fsdp2_full_reshard, fsdp2_full_noreshard, fsdp2_hybrid_reshard, fsdp2_hybrid_noreshard

set -euo pipefail

GYPSUM_DIR="${GYPSUM_DIR:-$HOME/projects/gypsum}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/run_nanotron_benchmark.sh"

# Defaults
DRY_RUN=0
MODELS="1b,3b"
STRATEGIES="ddp,zero1"
NODE_COUNTS="1,2,4,8"
EXPERIMENT_MODE=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1; shift ;;
        --smoke-test) NODE_COUNTS="1"; shift ;;
        --models)     MODELS="$2"; shift 2 ;;
        --strategies) STRATEGIES="$2"; shift 2 ;;
        --nodes)      NODE_COUNTS="$2"; shift 2 ;;
        --tp)         EXPERIMENT_MODE="tp"; shift ;;
        --pp)         EXPERIMENT_MODE="pp"; shift ;;
        --smol)       EXPERIMENT_MODE="smol"; shift ;;
        --fsdp)       EXPERIMENT_MODE="fsdp"; shift ;;
        *)            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Override defaults for experiment modes
if [[ "$EXPERIMENT_MODE" == "tp" ]]; then
    MODELS="8b"
    STRATEGIES="zero1"  # DDP OOMs for 8B even with recompute_layer=true
    EXTRA_ARGS="--tp-size 2 --seq-len 2048"
    echo "=== TP=2 Experiment Mode (LLaMA 8B, ZeRO-1 only, mbs=1, seq=2048, recompute_layer) ==="
elif [[ "$EXPERIMENT_MODE" == "pp" ]]; then
    MODELS="8b"
    STRATEGIES="zero1"  # DDP OOMs for 8B even with recompute_layer=true
    EXTRA_ARGS="--pp-size 2 --seq-len 2048"
    echo "=== PP=2 Experiment Mode (LLaMA 8B, ZeRO-1 only, mbs=1, seq=2048, recompute_layer) ==="
elif [[ "$EXPERIMENT_MODE" == "smol" ]]; then
    MODELS="smol3b"
    # smol3b forces TP=2, mbs=3, seq=4096 internally (exact production recipe)
    echo "=== SmolLM3 3B Experiment Mode — Production Recipe (TP=2, mbs=3, seq=4096) ==="
elif [[ "$EXPERIMENT_MODE" == "fsdp" ]]; then
    STRATEGIES="fsdp2_full_reshard,fsdp2_full_noreshard,fsdp2_hybrid_reshard,fsdp2_hybrid_noreshard"
    echo "=== FSDP2 Experiment Mode (4 FSDP variants) ==="
fi

# Convert comma-separated to arrays
IFS=',' read -ra MODEL_ARR <<< "$MODELS"
IFS=',' read -ra STRATEGY_ARR <<< "$STRATEGIES"
IFS=',' read -ra NODE_ARR <<< "$NODE_COUNTS"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "=== DRY RUN (no jobs will be submitted) ==="
    echo ""
fi

echo "Benchmark matrix:"
echo "  Models:     ${MODEL_ARR[*]}"
echo "  Strategies: ${STRATEGY_ARR[*]}"
echo "  Nodes:      ${NODE_ARR[*]}"
[[ -n "$EXTRA_ARGS" ]] && echo "  Extra args: $EXTRA_ARGS"
echo ""

submitted=0
skipped=0
for model in "${MODEL_ARR[@]}"; do
    for strategy in "${STRATEGY_ARR[@]}"; do
        for nodes in "${NODE_ARR[@]}"; do
            # Build job name
            if [[ -n "$EXPERIMENT_MODE" ]]; then
                job_name="nanotron_${EXPERIMENT_MODE}_${model}_${strategy}_${nodes}N"
            else
                job_name="nanotron_${model}_${strategy}_${nodes}N"
            fi

            echo "Submitting: $job_name (model=$model, strategy=$strategy, nodes=$nodes${EXTRA_ARGS:+, $EXTRA_ARGS})"

            if [[ $DRY_RUN -eq 0 ]]; then
                cd "$GYPSUM_DIR"
                GYPSUM_DIR="$GYPSUM_DIR" scripts/sbatch.sh \
                    --nodes "$nodes" \
                    --devices 8 \
                    --custom-script \
                    --n-retries 0 \
                    -J "$job_name" \
                    "$BENCHMARK_SCRIPT" \
                    --model-size "$model" \
                    --strategy "$strategy" \
                    $EXTRA_ARGS
                echo "  -> Submitted"
                sleep 2
            fi

            submitted=$((submitted + 1))
        done
    done
done

echo ""
echo "=== Total: $submitted jobs submitted, $skipped skipped ==="
if [[ $DRY_RUN -eq 1 ]]; then
    echo "(dry run -- no jobs were actually submitted)"
fi
