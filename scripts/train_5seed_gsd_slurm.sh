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
#SBATCH --job-name=gsd_seed
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/gsd_seed_%A_%a.out
#SBATCH --error=logs/gsd_seed_%A_%a.err
#SBATCH --array=42-46

# H1: 5-seed GSD training
# Each array task uses a separate experiment config with its own seed.
# Usage: sbatch scripts/train_5seed_gsd_slurm.sh

SEED=${SLURM_ARRAY_TASK_ID}
EXPERIMENT="exp_d_gsd_seed${SEED}"

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/VoiceFM_repro3
mkdir -p logs

echo "=== H1: GSD 5-Seed Training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Seed: ${SEED}"
echo "Experiment: ${EXPERIMENT}"
echo "================================"

python scripts/train.py --experiment ${EXPERIMENT}

echo ""
echo "=== Training complete: seed ${SEED} ==="
echo "Date: $(date)"
