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
#SBATCH --job-name=fewshot_v3
#SBATCH --output=logs/fewshot_v3_seed%a_%A.out
#SBATCH --error=logs/fewshot_v3_seed%a_%A.err
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=42-46

# H2-v3: Few-shot external-dataset eval for v3-retrained VoiceFM-HuBERT (exp_d) checkpoints.
# Reuses the v2 script body; only the --checkpoint path points at v3 seeds.
# Usage: sbatch scripts/evaluate_fewshot_v3_slurm.sh

SEED=${SLURM_ARRAY_TASK_ID}
EXPERIMENT="exp_d_gsd_v3_seed${SEED}"
CKPT_DIR="checkpoints_${EXPERIMENT}"

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM
mkdir -p logs
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=== H2-v3 few-shot (seed ${SEED}) ==="
echo "Date: $(date)"
echo "Checkpoint: ${CKPT_DIR}/best_model.pt"

python -u scripts/evaluate_fewshot_v3.py \
  --checkpoint ${CKPT_DIR}/best_model.pt \
  --experiment ${EXPERIMENT}

echo "=== Done: seed ${SEED} ==="
