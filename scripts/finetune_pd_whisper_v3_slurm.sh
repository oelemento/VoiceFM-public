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
#SBATCH --job-name=pd_whisper_v3
#SBATCH --output=logs/finetune_pd_whisper_v3_%j.out
#SBATCH --error=logs/finetune_pd_whisper_v3_%j.out
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM
export PYTHONPATH="$PWD:$PYTHONPATH"

SEED=${1:-43}
CHECKPOINT="checkpoints_exp_whisper_ft4_gsd_v3_seed${SEED}/best_model.pt"

echo "=== VoiceFM-Whisper PD Fine-tuning v3 (seed ${SEED}) ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Started: $(date)"

python -u scripts/finetune_pd.py \
    --init whisper-voicefm \
    --checkpoint "$CHECKPOINT" \
    --metadata data/mpower/mpower_metadata_dual.csv \
    --audio-dir data/mpower/audio \
    --seed "$SEED" \
    --out-dir results_v3/mpower_pd_whisper \
    --epochs 30 \
    --batch-size 8 \
    --grad-accum 8 \
    --lr-backbone 1e-5 \
    --lr-head 1e-4 \
    --patience 10 \
    --num-workers 4 \
    --task-type-map '{"sustained":786,"countdown":2}' \
    --skip-frozen-probe

echo "=== Done: $(date) ==="
