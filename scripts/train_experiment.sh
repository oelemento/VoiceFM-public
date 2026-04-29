#!/bin/bash
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00

# Usage: sbatch --job-name=voicefm_exp_a --output=logs/exp_a_%j.out --error=logs/exp_a_%j.err scripts/train_experiment.sh exp_a_large_batch
# Or use scripts/launch_experiments.sh to submit all at once.

EXPERIMENT=${1:?Usage: train_experiment.sh <experiment_name>}

# Load modules and activate environment
source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

# Project directory
cd ~/VoiceFM

# Create logs directory
mkdir -p logs

# Print environment info
echo "=== Experiment: ${EXPERIMENT} ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "================================="

# Run training with experiment config
python scripts/train.py --experiment "${EXPERIMENT}"
