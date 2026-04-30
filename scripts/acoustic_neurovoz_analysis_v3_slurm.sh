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
#SBATCH --job-name=nv_acoustic_v3
#SBATCH --output=logs/nv_acoustic_v3_%j.out
#SBATCH --error=logs/nv_acoustic_v3_%j.out
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM
export PYTHONPATH="$PWD:$PYTHONPATH"

CHECKPOINT="checkpoints_exp_d_gsd_v3_seed42/best_model.pt"

echo "=== NeuroVoz Acoustic Analysis (v3) ==="
echo "Checkpoint: ${CHECKPOINT}"

python scripts/acoustic_neurovoz_analysis_v3.py \
    --checkpoint "$CHECKPOINT" \
    --batch-size 8 \
    --out-dir results_v3/neurovoz
