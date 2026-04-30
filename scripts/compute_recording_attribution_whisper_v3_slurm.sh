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
#SBATCH --job-name=rec_attr_whisper_v3
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rec_attr_whisper_v3_seed%a_%A.out
#SBATCH --error=logs/rec_attr_whisper_v3_seed%a_%A.err
#SBATCH --array=42-46

# Recording attribution analysis (Fig 4a/4b) — Whisper v3 checkpoints, 5 seeds.
# Outputs to results_v3/ so it does not clobber v2 artifacts.
# Usage: sbatch scripts/compute_recording_attribution_whisper_v3_slurm.sh

SEED=${SLURM_ARRAY_TASK_ID}
EXPERIMENT="exp_whisper_ft4_gsd_v3_seed${SEED}"
CKPT="checkpoints_${EXPERIMENT}/best_model.pt"

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/VoiceFM
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p logs results_v3

echo "=== Recording Attribution v3 Whisper (seed ${SEED}) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Checkpoint: ${CKPT}"
echo "========================================================"

python scripts/compute_recording_attribution_v3.py \
    --checkpoint "${CKPT}" \
    --experiment "${EXPERIMENT}" \
    --baseline \
    --output "results_v3/recording_attribution_whisper_seed${SEED}.json"

echo ""
echo "=== Done: seed ${SEED} ==="
echo "Date: $(date)"
