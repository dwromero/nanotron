#!/bin/bash
# Nanotron benchmark launcher for SLURM (Together cluster)
# Runs inside Apptainer via gypsum's sbatch.sh -> run.sh pipeline.
#
# Usage (via gypsum sbatch.sh):
#   cd $GYPSUM_DIR && GYPSUM_DIR=$GYPSUM_DIR scripts/sbatch.sh \
#     --nodes 2 --devices 8 --custom-script \
#     /path/to/run_nanotron_benchmark.sh \
#     --model-size 3b --strategy ddp
#
# Arguments:
#   --model-size   : 1b, 3b, 8b, or smol3b (LLaMA / SmolLM3 architecture)
#   --strategy     : ddp, zero1
#   --tp-size N    : tensor parallel size (default: 1)
#   --pp-size N    : pipeline parallel size (default: 1)
#   --seq-len N    : override default seq_len for the model
#   --train-steps N: override default train steps (default: 650)

set -e

NANOTRON_DIR="${NANOTRON_DIR:-$(cd "$(dirname "$0")" && pwd)}"

# ============================================================================
# Sentinel-based coordination: only one torchrun per node
# ============================================================================
SENTINEL="/tmp/nanotron_done_${SLURM_JOB_ID}_${SLURM_NODEID}"
LOCAL_ID=${SLURM_LOCALID:-0}
if [[ "$LOCAL_ID" != "0" ]]; then
    echo "Task $LOCAL_ID: Waiting for torchrun to finish..."
    while [[ ! -f "$SENTINEL" ]]; do
        sleep 2
    done
    echo "Task $LOCAL_ID: Torchrun finished, exiting"
    exit 0
fi
rm -f "$SENTINEL"

# ============================================================================
# Parse arguments
# ============================================================================
MODEL_SIZE=""
STRATEGY=""
TP_SIZE=1
PP_SIZE=1
CP_SIZE=1
SEQ_LEN_OVERRIDE=""
MBS_OVERRIDE=""
TRAIN_STEPS=650
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-size) MODEL_SIZE="$2"; shift 2 ;;
        --strategy)   STRATEGY="$2"; shift 2 ;;
        --tp-size)    TP_SIZE="$2"; shift 2 ;;
        --pp-size)    PP_SIZE="$2"; shift 2 ;;
        --cp-size)    CP_SIZE="$2"; shift 2 ;;
        --seq-len)    SEQ_LEN_OVERRIDE="$2"; shift 2 ;;
        --mbs)        MBS_OVERRIDE="$2"; shift 2 ;;
        --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_SIZE" || -z "$STRATEGY" ]]; then
    echo "Usage: $0 --model-size {1b,3b,8b,smol3b} --strategy {ddp,zero1,fsdp2_full_reshard,fsdp2_full_noreshard,fsdp2_hybrid_reshard,fsdp2_hybrid_noreshard} [--tp-size N] [--pp-size N] [--cp-size N]"
    exit 1
fi

# ============================================================================
# Detect SLURM environment
# ============================================================================
echo "SLURM env:"
echo "  SLURM_NODELIST=$SLURM_NODELIST"
echo "  SLURM_NNODES=$SLURM_NNODES"
echo "  SLURM_NODEID=$SLURM_NODEID"

# Parse master address from SLURM
if [[ -n "$SLURM_NODELIST" ]]; then
    if [[ "$SLURM_NODELIST" =~ ^([^[]+)\[([0-9]+) ]]; then
        MASTER_ADDR="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    elif [[ "$SLURM_NODELIST" =~ ^([^,]+), ]]; then
        MASTER_ADDR="${BASH_REMATCH[1]}"
    else
        MASTER_ADDR="$SLURM_NODELIST"
    fi
    export MASTER_ADDR
    echo "Parsed MASTER_ADDR=$MASTER_ADDR"
fi

export MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${SLURM_NNODES:-1}
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# ============================================================================
# Strategy -> config mapping
# ============================================================================
ZERO_STAGE=0
DP_ENGINE="ddp"
FSDP_RESHARD="true"
FSDP_HYBRID="false"

case "$STRATEGY" in
    ddp)
        ZERO_STAGE=0
        DP_ENGINE="ddp"
        ;;
    zero1)
        ZERO_STAGE=1
        DP_ENGINE="ddp"
        ;;
    fsdp2_full_reshard)
        ZERO_STAGE=0
        DP_ENGINE="fsdp2"
        FSDP_RESHARD="true"
        FSDP_HYBRID="false"
        ;;
    fsdp2_full_noreshard)
        ZERO_STAGE=0
        DP_ENGINE="fsdp2"
        FSDP_RESHARD="false"
        FSDP_HYBRID="false"
        ;;
    fsdp2_hybrid_reshard)
        ZERO_STAGE=0
        DP_ENGINE="fsdp2"
        FSDP_RESHARD="true"
        FSDP_HYBRID="true"
        ;;
    fsdp2_hybrid_noreshard)
        ZERO_STAGE=0
        DP_ENGINE="fsdp2"
        FSDP_RESHARD="false"
        FSDP_HYBRID="true"
        ;;
    *)
        echo "ERROR: Unknown strategy '$STRATEGY'. Must be one of: ddp, zero1, fsdp2_full_reshard, fsdp2_full_noreshard, fsdp2_hybrid_reshard, fsdp2_hybrid_noreshard."
        exit 1
        ;;
esac

# ============================================================================
# Model configurations (matching S4/S5 for comparison)
# ============================================================================
case "$MODEL_SIZE" in
    1b)
        HIDDEN_SIZE=2048
        INTERMEDIATE_SIZE=8192
        NUM_LAYERS=16
        NUM_HEADS=32
        NUM_KV_HEADS=8
        SEQ_LEN=${SEQ_LEN_OVERRIDE:-4096}
        MAX_POS_EMB=4096
        MBS=2
        IS_LLAMA=true
        VOCAB_SIZE=128256
        ;;
    3b)
        HIDDEN_SIZE=3072
        INTERMEDIATE_SIZE=8192
        NUM_LAYERS=28
        NUM_HEADS=24
        NUM_KV_HEADS=8
        SEQ_LEN=${SEQ_LEN_OVERRIDE:-2048}
        MAX_POS_EMB=131072
        MBS=1
        IS_LLAMA=true
        VOCAB_SIZE=128256
        ;;
    8b)
        HIDDEN_SIZE=4096
        INTERMEDIATE_SIZE=14336
        NUM_LAYERS=32
        NUM_HEADS=32
        NUM_KV_HEADS=8
        SEQ_LEN=${SEQ_LEN_OVERRIDE:-4096}
        MAX_POS_EMB=131072
        MBS=1
        IS_LLAMA=true
        VOCAB_SIZE=128256
        ;;
    smol3b)
        # SmolLM3 3B — exact HuggingFace production recipe (stage1_8T.yaml)
        # TP=2, mbs=3, seq=4096, zero_stage=0, accumulate_grad_in_fp32=true
        # rope_theta=50000, no_rope_layer=4, rms_norm_eps=1e-6, ddp_bucket_cap_mb=50
        HIDDEN_SIZE=2048
        INTERMEDIATE_SIZE=11008
        NUM_LAYERS=36
        NUM_HEADS=16
        NUM_KV_HEADS=4
        SEQ_LEN=${SEQ_LEN_OVERRIDE:-4096}
        MAX_POS_EMB=4096
        MBS=3
        IS_LLAMA=false  # uses is_qwen2_config (like production)
        IS_SMOL=true
        VOCAB_SIZE=128256
        RMS_NORM_EPS="1.0e-06"
        ROPE_THETA=50000.0
        DDP_BUCKET_CAP_MB=50
        # Force TP=2 to match production recipe (override any CLI --tp-size)
        TP_SIZE=2
        ;;
    *)
        echo "ERROR: Unknown model size '$MODEL_SIZE'. Must be 1b, 3b, 8b, or smol3b."
        exit 1
        ;;
esac

# Apply MBS override if provided
if [[ -n "$MBS_OVERRIDE" ]]; then
    MBS="$MBS_OVERRIDE"
fi

# Defaults for optional model-specific variables
IS_SMOL=${IS_SMOL:-false}
RMS_NORM_EPS=${RMS_NORM_EPS:-"1.0e-05"}
ROPE_THETA=${ROPE_THETA:-10000.0}
DDP_BUCKET_CAP_MB=${DDP_BUCKET_CAP_MB:-25}

# ============================================================================
# Compute DP size
# ============================================================================
DP_SIZE=$((WORLD_SIZE / (TP_SIZE * PP_SIZE * CP_SIZE)))
if [[ $DP_SIZE -lt 1 ]]; then
    echo "ERROR: DP_SIZE=$DP_SIZE (WORLD_SIZE=$WORLD_SIZE, TP=$TP_SIZE, PP=$PP_SIZE, CP=$CP_SIZE). Not enough GPUs."
    exit 1
fi

# ============================================================================
# Generate YAML config
# ============================================================================
CONFIG_DIR="/tmp/nanotron_configs"
mkdir -p "$CONFIG_DIR"
JOB_ID=${SLURM_JOB_ID:-$$}
CONFIG_FILE="${CONFIG_DIR}/config_${MODEL_SIZE}_${STRATEGY}_tp${TP_SIZE}_pp${PP_SIZE}_cp${CP_SIZE}_${NNODES}N_${JOB_ID}.yaml"
CHECKPOINT_DIR="/tmp/nanotron_checkpoints_${JOB_ID}"
BENCHMARK_CSV="/tmp/nanotron_benchmark_${JOB_ID}.csv"

# Determine model config type
if [[ "$IS_LLAMA" == "true" ]]; then
    MODEL_CONFIG_TYPE="is_llama_config: true"
else
    MODEL_CONFIG_TYPE="is_qwen2_config: true"
fi

# SmolLM3-specific extra model config fields (production recipe)
SMOL_EXTRA_MODEL_CONFIG=""
if [[ "$IS_SMOL" == "true" ]]; then
    SMOL_EXTRA_MODEL_CONFIG="
    no_rope_layer: 4
    _fused_rms_norm: true
    _fused_rotary_emb: true
    _use_qkv_packed: true
    _use_doc_masking: false
    attention_bias: false
    rope_interleaved: false
    z_loss_enabled: false
    z_loss_coefficient: 1.0e-05"
fi

# PP engine
PP_ENGINE="1f1b"

# Activation checkpointing: enable for 8B models with TP or PP > 1 (DDP OOMs otherwise)
RECOMPUTE_LAYER="false"
if [[ "$MODEL_SIZE" == "8b" && ( "$TP_SIZE" -gt 1 || "$PP_SIZE" -gt 1 ) ]]; then
    RECOMPUTE_LAYER="true"
fi

# FP32 gradient accumulation: disable for ZeRO-1 (DDP comm hook reduce_scatter is
# NotImplementedError) and for FSDP2 (manages its own gradient lifecycle)
ACCUMULATE_FP32="true"
if [[ "$ZERO_STAGE" -gt 0 || "$DP_ENGINE" == "fsdp2" ]]; then
    ACCUMULATE_FP32="false"
fi

cat > "$CONFIG_FILE" << YAML_EOF
checkpoints:
  checkpoint_interval: 999999
  checkpoints_path: ${CHECKPOINT_DIR}
  checkpoints_path_is_shared_file_system: false
  resume_checkpoint_path: null
  save_initial_state: false
data_stages:
- data:
    dataset: null
    num_loading_workers: 1
    seed: 42
  name: Benchmark Stage
  start_training_step: 1
general:
  benchmark_csv_path: ${BENCHMARK_CSV}
  consumed_train_samples: null
  ignore_sanity_checks: true
  project: fsdp-bench
  run: nanotron_${MODEL_SIZE}_${STRATEGY}_tp${TP_SIZE}_pp${PP_SIZE}_${NNODES}N
  seed: 42
  step: null
lighteval: null
logging:
  iteration_step_info_interval: 1
  log_level: info
  log_level_replica: info
model:
  ddp_bucket_cap_mb: ${DDP_BUCKET_CAP_MB}
  dtype: bfloat16
  init_method:
    std: 0.02
  make_vocab_size_divisible_by: 1
  model_config:
    bos_token_id: 128000
    eos_token_id: 128001
    hidden_act: silu
    hidden_size: ${HIDDEN_SIZE}
    initializer_range: 0.02
    intermediate_size: ${INTERMEDIATE_SIZE}
    ${MODEL_CONFIG_TYPE}
    max_position_embeddings: ${MAX_POS_EMB}
    num_attention_heads: ${NUM_HEADS}
    num_hidden_layers: ${NUM_LAYERS}
    num_key_value_heads: ${NUM_KV_HEADS}
    pad_token_id: null
    pretraining_tp: ${TP_SIZE}
    rms_norm_eps: ${RMS_NORM_EPS}
    rope_scaling: null
    rope_theta: ${ROPE_THETA}
    tie_word_embeddings: true
    use_cache: true
    vocab_size: ${VOCAB_SIZE}${SMOL_EXTRA_MODEL_CONFIG}
optimizer:
  accumulate_grad_in_fp32: ${ACCUMULATE_FP32}
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: 0.0003
    lr_decay_starting_step: null
    lr_decay_steps: ${TRAIN_STEPS}
    lr_decay_style: cosine
    lr_warmup_steps: 10
    lr_warmup_style: linear
    min_decay_lr: 1.0e-05
  optimizer_factory:
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    name: adamW
    torch_adam_is_fused: true
  weight_decay: 0.01
  zero_stage: ${ZERO_STAGE}
parallelism:
  dp: ${DP_SIZE}
  dp_engine: ${DP_ENGINE}
  fsdp_reshard_after_forward: ${FSDP_RESHARD}
  fsdp_hybrid: ${FSDP_HYBRID}
  context_parallel_size: ${CP_SIZE}
  expert_parallel_size: 1
  pp: ${PP_SIZE}
  pp_engine: ${PP_ENGINE}
  tp: ${TP_SIZE}
  tp_linear_async_communication: true
  tp_mode: REDUCE_SCATTER
  tp_recompute_allgather: true
  recompute_layer: ${RECOMPUTE_LAYER}
profiler: null
tokenizer:
  tokenizer_max_length: null
  tokenizer_name_or_path: robot-test/dummy-tokenizer-wordlevel
  tokenizer_revision: null
tokens:
  batch_accumulation_per_replica: 1
  limit_test_batches: 0
  limit_val_batches: 0
  micro_batch_size: ${MBS}
  sequence_length: ${SEQ_LEN}
  train_steps: ${TRAIN_STEPS}
  val_check_interval: -1
YAML_EOF

# ============================================================================
# Print launch configuration
# ============================================================================
echo "============================================================"
echo "  Nanotron Benchmark"
echo "============================================================"
echo "Model size: $MODEL_SIZE"
echo "Strategy: $STRATEGY (zero_stage=$ZERO_STAGE, dp_engine=$DP_ENGINE)"
echo "  TP=$TP_SIZE, PP=$PP_SIZE, CP=$CP_SIZE"
echo "  DP=$DP_SIZE (computed)"
echo "  World size: $WORLD_SIZE (${NNODES}N x ${NPROC_PER_NODE} GPUs)"
echo "  Micro batch size: $MBS"
echo "  Sequence length: $SEQ_LEN"
echo "  Train steps: $TRAIN_STEPS"
echo "  Recompute layer (AC): $RECOMPUTE_LAYER"
echo "  Accumulate grad FP32: $ACCUMULATE_FP32"
if [[ "$DP_ENGINE" == "fsdp2" ]]; then
echo "  FSDP reshard_after_forward: $FSDP_RESHARD"
echo "  FSDP hybrid (HSDP): $FSDP_HYBRID"
fi
echo "  Config: $CONFIG_FILE"
echo "============================================================"

# ============================================================================
# Install Nanotron (if needed)
# ============================================================================
if ! python -c "import nanotron" 2>/dev/null; then
    echo "=== Installing Nanotron ==="
    pip install -e "${NANOTRON_DIR}[all]" 2>&1 | tail -5
fi

# grouped_gemm is a required dependency for Nanotron's MoE module (imported unconditionally)
if ! python -c "import grouped_gemm" 2>/dev/null; then
    echo "=== Installing grouped_gemm ==="
    pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main 2>&1 | tail -5
fi

# ============================================================================
# Launch training
# ============================================================================
LAUNCH_TS=$(date +%s.%N)
echo "NANOTRON_LAUNCH_TIMESTAMP=$LAUNCH_TS"

EXIT_CODE=0
CUDA_DEVICE_MAX_CONNECTIONS=1 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=$JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500} \
    "${NANOTRON_DIR}/run_train.py" \
    --config-file "$CONFIG_FILE" \
|| EXIT_CODE=$?

FINISH_TS=$(date +%s.%N)
TOTAL_WALL=$(python3 -c "print(f'{$FINISH_TS - $LAUNCH_TS:.2f}')")
echo "NANOTRON_FINISH_TIMESTAMP=$FINISH_TS"
echo "NANOTRON_TOTAL_WALL_SECONDS=$TOTAL_WALL"

# ============================================================================
# Cleanup
# ============================================================================
rm -rf "$CHECKPOINT_DIR" 2>/dev/null || true

# Signal sentinel for other tasks on this node
echo "Task 0: Created sentinel $SENTINEL, exiting with code $EXIT_CODE"
touch "$SENTINEL"

exit $EXIT_CODE
