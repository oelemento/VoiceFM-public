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
#SBATCH --job-name=fig4_attr_v3
#SBATCH --output=logs/fig4_attr_v3_%j.out
#SBATCH --error=logs/fig4_attr_v3_%j.out
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

# Pure plotting script: reads cached JSONs in results_v3/ and writes paper_v3/
# No GPU, no extraction. Defer running until Phase 4 (figure regeneration).

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=== Fig 4 attribution (v3) ==="
python scripts/paper_fig4_attribution_v3.py
