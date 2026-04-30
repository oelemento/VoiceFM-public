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
#SBATCH --job-name=neurovoz_v3
#SBATCH --output=logs/neurovoz_v3_seed%a_%A.out
#SBATCH --error=logs/neurovoz_v3_seed%a_%A.out
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=42-46

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM
export PYTHONPATH="$PWD:$PYTHONPATH"

SEED=${SLURM_ARRAY_TASK_ID}
CHECKPOINT="checkpoints_exp_d_gsd_v3_seed${SEED}/best_model.pt"

echo "=== NeuroVoz eval v3 (seed ${SEED}) ==="
echo "Checkpoint: ${CHECKPOINT}"

python scripts/evaluate_neurovoz_v3.py \
    --checkpoint "$CHECKPOINT" \
    --batch-size 8 \
    --n-trials 100
