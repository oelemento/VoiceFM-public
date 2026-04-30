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
#SBATCH --job-name=h29v3-wmt
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/h29v3_whisper_v3_mt_seed%a_%A.out
#SBATCH --error=logs/h29v3_whisper_v3_mt_seed%a_%A.err
#SBATCH --array=42-46

# H29v3: Whisper large-v3 multi-task classification (1M+ hrs training data, 128 mel bins)
# Usage: sbatch scripts/h29v3_whisper_v3_mt_gsd_slurm.sh

SEED=${SLURM_ARRAY_TASK_ID}
EXPERIMENT="exp_whisper_v3_mt_gsd_seed${SEED}"
CKPT_DIR="checkpoints_${EXPERIMENT}"

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/VoiceFM
mkdir -p logs
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=== H29v3: Whisper v3 Multi-Task GSD (seed ${SEED}) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# --- Train ---
python -u scripts/train_whisper_multitask.py --experiment ${EXPERIMENT}
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${TRAIN_EXIT}"
    exit $TRAIN_EXIT
fi

echo ">>> Training complete: $(date)"

# --- Evaluate ---
python -u scripts/evaluate_whisper_multitask.py \
    --checkpoint ${CKPT_DIR}/best_model.pt \
    --experiment ${EXPERIMENT}
EVAL_EXIT=$?

echo "=== Done: seed ${SEED}, train=${TRAIN_EXIT}, eval=${EVAL_EXIT} ==="
