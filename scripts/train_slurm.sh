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
#SBATCH --job-name=voicefm
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/voicefm_%j.out
#SBATCH --error=logs/voicefm_%j.err

# Load modules and activate environment
source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

# HuggingFace token (needed for gated models like HeAR)
# Set via: export HF_TOKEN=hf_xxx before sbatch, or in ~/.bashrc
if [ -z "$HF_TOKEN" ]; then
    # Try reading from huggingface cache
    HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; t=get_token(); print(t if t else '')" 2>/dev/null)
fi
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
    echo "HF_TOKEN: set"
fi

# CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Project directory
cd ~/VoiceFM

# Create logs directory
mkdir -p logs

# Print environment info
echo "=== Environment ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "==================="

# Run training — pass experiment name if provided
# Usage: sbatch scripts/train_slurm.sh [experiment_name]
# Example: sbatch scripts/train_slurm.sh exp_f_unfreeze6
if [ -n "$1" ]; then
    echo "Experiment: $1"
    python scripts/train.py --experiment "$1"
else
    echo "Experiment: baseline (no overrides)"
    python scripts/train.py
fi
