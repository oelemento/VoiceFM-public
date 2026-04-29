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
#SBATCH --job-name=mpower_download
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mpower_download_%j.out
#SBATCH --error=logs/mpower_download_%j.out

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM

# Ensure synapseclient is installed
pip install -q synapseclient

python scripts/download_mpower.py \
    --out-dir data/mpower
