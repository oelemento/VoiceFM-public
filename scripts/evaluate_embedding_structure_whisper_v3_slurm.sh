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
#SBATCH --job-name=embedding_structure_v3_whisper
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/embedding_structure_whisper_v3_%j.out
#SBATCH --error=logs/embedding_structure_whisper_v3_%j.out

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p results_v3 logs

python -u scripts/evaluate_embedding_structure_v3.py \
    --checkpoint checkpoints_exp_whisper_ft4_gsd_v3_seed42/best_model.pt \
    --out-dir results_v3/embedding_structure_whisper
