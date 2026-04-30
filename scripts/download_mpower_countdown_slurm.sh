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
#SBATCH --job-name=mpower_cd
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mpower_countdown_%j.out
#SBATCH --error=logs/mpower_countdown_%j.out

# Download mPower countdown audio files from Synapse (individual downloads)

set -e

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM

python3.11 scripts/download_mpower_countdown.py --out-dir data/mpower
