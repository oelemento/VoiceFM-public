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
#SBATCH --job-name=convert_m4a
#SBATCH --partition=scu-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=logs/convert_m4a_%j.out
#SBATCH --error=logs/convert_m4a_%j.out

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM

echo "Starting m4a → wav conversion at $(date)"
python3.11 scripts/convert_m4a_to_wav.py \
    --audio-dir data/mpower/audio \
    --workers 8

echo "Conversion finished at $(date)"
