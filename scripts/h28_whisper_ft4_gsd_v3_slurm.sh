#!/bin/bash
# ============================================================================
# CLUSTER-SPECIFIC TEMPLATE
# ----------------------------------------------------------------------------
# This script targets the Cornell Cayuga HPC cluster (NVIDIA A40/A100 nodes,
# SLURM 25.05+, scu-gpu partition). It will NOT run unmodified elsewhere.
# Before submitting, edit:
#   - #SBATCH --partition=...   to match your cluster's partition
#   - the `module load`         block (Cayuga uses lmod + mamba)
#   - the `source activate`     line (replace with your venv/conda env)
#   - the `cd ...` line          to point at your project root
# ============================================================================
#SBATCH --job-name=h28-wft4-v3
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/h28_whisper_ft4_v3_seed%a_%A.out
#SBATCH --error=logs/h28_whisper_ft4_v3_seed%a_%A.err
#SBATCH --array=42-46

# H28-v3: Whisper large-v2 VoiceFM on corrected v3.0.0 labels (846 train).
# Same model, same loss, same seeds as v2.3.0 H28; only --data-dir changes.
# Usage: sbatch scripts/h28_whisper_ft4_gsd_v3_slurm.sh

SEED=${SLURM_ARRAY_TASK_ID}
EXPERIMENT="exp_whisper_ft4_gsd_v3_seed${SEED}"
CKPT_DIR="checkpoints_${EXPERIMENT}"
DATA_DIR="data/processed_v3"

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/VoiceFM
mkdir -p logs
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=== H28-v3: Whisper FT4 VoiceFM GSD v3 (seed ${SEED}) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Data dir: ${DATA_DIR}"

# --- Train ---
python -u scripts/train.py --experiment ${EXPERIMENT} --data-dir ${DATA_DIR}
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${TRAIN_EXIT}"
    exit $TRAIN_EXIT
fi

# --- Evaluate ---
python -u scripts/evaluate.py \
    --checkpoint ${CKPT_DIR}/best_model.pt \
    --experiment ${EXPERIMENT} \
    --data-dir ${DATA_DIR} \
    --baseline \
    --no-umap
EVAL_EXIT=$?

echo "=== Done: seed ${SEED}, exit=${EVAL_EXIT} ==="
