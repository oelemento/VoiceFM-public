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
#SBATCH --job-name=h27_v3_neurovoz
#SBATCH --output=logs/eval_h27_v3_neurovoz_%j.out
#SBATCH --error=logs/eval_h27_v3_neurovoz_%j.out
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=== H27 eval on neurovoz ==="
python -u scripts/eval_h27_external_v3.py --dataset neurovoz
echo "=== Done ==="
