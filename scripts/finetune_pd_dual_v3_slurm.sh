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
#SBATCH --job-name=pd_dual_v3
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=1-24
#SBATCH --output=logs/pd_dual_v3_%A_%a.out
#SBATCH --error=logs/pd_dual_v3_%A_%a.out

# H24: Fine-tune PD classifier with dual recording types (phonation + countdown)

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM

METADATA="data/mpower/mpower_metadata_dual.csv"
AUDIO_DIR="data/mpower/audio"
OUT_DIR="results_v3/mpower_pd_dual"

# Map array task ID to (init, max_participants, seed)
# 24 runs: 2 inits × 4 regimes × 3 seeds
TASK_ID=$SLURM_ARRAY_TASK_ID

# Decode task ID
INIT_IDX=$(( (TASK_ID - 1) / 12 ))  # 0=voicefm, 1=hubert
REMAINDER=$(( (TASK_ID - 1) % 12 ))
REGIME_IDX=$(( REMAINDER / 3 ))      # 0=full, 1=500, 2=100, 3=50
SEED_IDX=$(( REMAINDER % 3 ))        # 0=42, 1=43, 2=44

if [ $INIT_IDX -eq 0 ]; then
    INIT="voicefm"
else
    INIT="hubert"
fi

REGIMES=(0 500 100 50)
MAX_PARTICIPANTS=${REGIMES[$REGIME_IDX]}

SEEDS=(42 43 44)
SEED=${SEEDS[$SEED_IDX]}

CHECKPOINT="checkpoints_exp_d_gsd_v3_seed${SEED}/best_model.pt"

echo "=== Task $TASK_ID: init=$INIT, max_participants=$MAX_PARTICIPANTS, seed=$SEED ==="

CMD="python scripts/finetune_pd.py \
    --init $INIT \
    --metadata $METADATA \
    --audio-dir $AUDIO_DIR \
    --max-train-participants $MAX_PARTICIPANTS \
    --seed $SEED \
    --out-dir $OUT_DIR \
    --experiment exp_d_gsd"

if [ "$INIT" = "voicefm" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

echo "Running: $CMD"
eval $CMD

echo "=== Done: Task $TASK_ID ==="
echo "Date: $(date)"
