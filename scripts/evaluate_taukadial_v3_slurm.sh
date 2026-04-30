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
#SBATCH --job-name=taukadial_v3
#SBATCH --output=logs/taukadial_v3_%j.out
#SBATCH --error=logs/taukadial_v3_%j.out
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=== TAUKADIAL evaluation v3 (frozen Whisper) ==="
python -u scripts/evaluate_taukadial_v3.py
echo "=== Done ==="
