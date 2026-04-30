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
#SBATCH --job-name=attention_saliency_v3
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/attention_saliency_v3_%j.out
#SBATCH --error=logs/attention_saliency_v3_%j.out

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p results_v3 logs

python scripts/evaluate_attention_saliency_v3.py \
    --checkpoint checkpoints_exp_d_gsd_v3_seed42/best_model.pt \
    --capev-scores data/processed/capev_scores.parquet \
    --baseline \
    --out-dir results_v3/attention_saliency
