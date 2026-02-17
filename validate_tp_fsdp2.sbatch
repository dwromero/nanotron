#!/bin/bash
#SBATCH --job-name=validate_tp_fsdp2
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/david.romero/projects/fsdp-bench/nanotron/validate_tp_fsdp2_%j.log

set -euo pipefail

NANOTRON_DIR="/home/david.romero/projects/fsdp-bench/nanotron"
GYPSUM_DIR="/home/david.romero/projects/gypsum"
SIF_IMAGE="/shared/images/gypsum_refactor_26.01.22.sif"

echo "=== Job $SLURM_JOB_ID on $(hostname), $(date) ==="
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Run inside Apptainer container (same as gypsum pipeline)
apptainer -c /shared/images/apptainer.conf run --nv --no-eval --writable-tmpfs \
    --bind /home:/home --bind /scratch:/scratch --bind /shared:/shared \
    "$SIF_IMAGE" \
    bash -c "
set -euo pipefail
cd $NANOTRON_DIR

# Install deps inside container
pip install -e '.[all]' 2>&1 | tail -3
pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main 2>&1 | tail -3

export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=disabled
export NANOTRON_LOG_PRECISION=high
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

echo ''
echo '=== Running TP=2 + FSDP2 validation (4 GPUs, bfloat16, 10 steps) ==='
python validate_fsdp2_nanotron.py --nproc 4 --tp 2 --steps 10 --dtype bfloat16

echo ''
echo '=== Also running standard DP-only validation (4 GPUs, bfloat16, 10 steps) ==='
python validate_fsdp2_nanotron.py --nproc 4 --steps 10 --dtype bfloat16

echo ''
echo '=== All validations complete ==='
"
