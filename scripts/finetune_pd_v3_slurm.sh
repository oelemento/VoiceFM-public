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
#SBATCH --job-name=pd_finetune_v3
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=1-30
#SBATCH --output=logs/pd_finetune_v3_%A_%a.out
#SBATCH --error=logs/pd_finetune_v3_%A_%a.out

source /etc/profile.d/lmod.sh
module load mamba
source activate voicefm

cd ~/VoiceFM

METADATA="data/mpower/mpower_metadata.csv"
AUDIO_DIR="data/mpower/audio"
OUT_DIR="results_v3/mpower_pd"

# Map array task ID to (init, max_participants, seed)
# 30 runs: 2 inits × 5 regimes × 3 seeds
TASK_ID=$SLURM_ARRAY_TASK_ID

# Decode task ID
INIT_IDX=$(( (TASK_ID - 1) / 15 ))  # 0=voicefm, 1=hubert
REMAINDER=$(( (TASK_ID - 1) % 15 ))
REGIME_IDX=$(( REMAINDER / 3 ))      # 0=full, 1=500, 2=200, 3=100, 4=50
SEED_IDX=$(( REMAINDER % 3 ))        # 0=42, 1=43, 2=44

if [ $INIT_IDX -eq 0 ]; then
    INIT="voicefm"
else
    INIT="hubert"
fi

REGIMES=(0 500 200 100 50)
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
