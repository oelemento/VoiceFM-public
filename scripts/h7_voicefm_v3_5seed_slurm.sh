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
#SBATCH --job-name=h7-vfm-v3
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/h7_vfm_v3_seed%a_%A.out
#SBATCH --error=logs/h7_vfm_v3_seed%a_%A.err
#SBATCH --array=42-46

# H7-v3: VoiceFM-HuBERT on corrected v3.0.0 labels (846 train).
# Same model, same loss, same seeds as v2.3.0 H7; only --data-dir changes.
# Usage: sbatch scripts/h7_voicefm_v3_5seed_slurm.sh

SEED=${SLURM_ARRAY_TASK_ID}
EXPERIMENT="exp_d_gsd_v3_seed${SEED}"
CKPT_DIR="checkpoints_${EXPERIMENT}"
DATA_DIR="data/processed_v3"

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/VoiceFM
mkdir -p logs

echo "=== H7-v3: VoiceFM-HuBERT 5-Seed (seed ${SEED}) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Experiment: ${EXPERIMENT}"
echo "Data dir: ${DATA_DIR}"
echo "=========================================="

# --- Train ---
echo ""
echo ">>> Phase 1: Training VoiceFM (seed ${SEED})"
python scripts/train.py --experiment ${EXPERIMENT} --data-dir ${DATA_DIR}
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${TRAIN_EXIT}"
    exit $TRAIN_EXIT
fi

echo ""
echo ">>> Training complete: $(date)"

# --- Evaluate ---
echo ""
echo ">>> Phase 2: Evaluating VoiceFM + HuBERT baseline (seed ${SEED})"
python scripts/evaluate.py \
    --checkpoint ${CKPT_DIR}/best_model.pt \
    --experiment ${EXPERIMENT} \
    --data-dir ${DATA_DIR} \
    --baseline \
    --no-umap
EVAL_EXIT=$?

if [ $EVAL_EXIT -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code ${EVAL_EXIT}"
    exit $EVAL_EXIT
fi

echo ""
echo "=== Done: VoiceFM seed ${SEED} ==="
echo "Date: $(date)"
echo "Results: ${CKPT_DIR}/eval_results_best_model.json"
