#!/bin/bash
# ============================================================================
# CLUSTER-SPECIFIC TEMPLATE
# ----------------------------------------------------------------------------
# This script targets your HPC cluster (GPU nodes,
# SLURM, cluster-specific partition). It will NOT run unmodified elsewhere.
# Before submitting, edit:
#   - #SBATCH --partition=...   to match your cluster's partition
#   - the `module load`         block (your cluster's module + conda setup)
#   - the `source activate`     line (replace with your venv/conda env)
#   - the `cd ...` line          to point at your project root
# ============================================================================
#SBATCH --job-name=pvqd-eval_v3
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pvqd_eval_v3_seed%a_%A.out
#SBATCH --error=logs/pvqd_eval_v3_seed%a_%A.err
#SBATCH --array=42-46

# Evaluate VoiceFM on PVQD: H24 (pathological detection), H25 (GRBAS), CAPE-V
# Each array task evaluates one v3 seed checkpoint.
# Usage: sbatch scripts/evaluate_pvqd_v3_slurm.sh

SEED=${SLURM_ARRAY_TASK_ID}
EXPERIMENT="exp_d_gsd_v3_seed${SEED}"
CKPT_DIR="checkpoints_${EXPERIMENT}"

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/VoiceFM
mkdir -p logs results_v3/seeds

echo "=== PVQD CAPE-V Evaluation v3 (seed ${SEED}) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Checkpoint: ${CKPT_DIR}/best_model.pt"
echo "=============================================="

python scripts/evaluate_pvqd_v3.py \
    --checkpoint ${CKPT_DIR}/best_model.pt \
    --experiment ${EXPERIMENT} \
    --baseline \
    --seed ${SEED} \
    --out-dir results_v3/seeds

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code ${EXIT_CODE}"
    exit $EXIT_CODE
fi

echo ""
echo "=== Done: PVQD eval v3 seed ${SEED} ==="
echo "Date: $(date)"
echo "Results: results_v3/seeds/pvqd_eval_seed${SEED}.json"
