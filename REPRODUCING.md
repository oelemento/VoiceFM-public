# Reproducing the VoiceFM paper

This guide walks through reproducing every result and figure in the paper, end-to-end, from raw audio to assembled figure PDF. There are three reproduction paths depending on how much of the pipeline you want to run yourself.

| Path | What you need | What you can reproduce |
|------|---------------|------------------------|
| **A. Figures only** | This repo's `results_v3/` JSONs | All paper figures, except S2/S3 (need embeddings) |
| **B. Embeddings + figures** | + cached embeddings (separate release) | All figures including S2/S3 and t-SNE |
| **C. Full pipeline** | + raw audio (B2AI Voice + external datasets) | Everything from training to figures |

## Path A — Figure-only reproduction

The `results_v3/` directory contains the result JSONs from the runs reported in the paper. You can regenerate most figures from these alone:

```bash
pip install -r requirements.txt

# Main figures
python scripts/paper_fig1b_training_v3.py            # Fig 1b,c training curves
python scripts/paper_fig2a_results_v3.py             # Fig 2a GSD AUROC bars
python scripts/paper_fig2b_diagnoses_v3.py           # Fig 2b per-diagnosis AUROC
python scripts/paper_fig3_prospective_v3.py          # Fig 3 prospective held-out cohort
python scripts/paper_fig3_transfer_v3.py             # Fig 4a external transfer
python scripts/paper_fig3b_fewshot_v3.py             # Fig 4b few-shot
python scripts/paper_fig4_attribution_v3.py          # Fig 5 recording attribution
python scripts/paper_fig5_composite_v3.py            # Fig 6 PD detection composite

# Supplementary figures
python scripts/paper_figS1_models_v3.py              # Fig S1 model comparison

# Assemble into a single PDF
python scripts/paper_assemble_pdf_v3.py
```

Figures S2 (acoustic grounding) and S3 (embedding structure) and the t-SNE figure require cached embedding files that are not in this repository. Download them from the separate data release (DOI in the paper) and place under `results_v3/`:

```
results_v3/
├── voicefm_whisper_embeddings.npz
├── voicefm_whisper_recording_embeddings.npz
└── ... (other npz)
```

Then:
```bash
python scripts/paper_figS2_interpretability_v3.py
python scripts/paper_figS3_embedding_structure_v3.py
python scripts/paper_figS4_tsne_v3.py
```

## Path B — Re-running embedding-derived analyses

If you have access to a pretrained VoiceFM-Whisper checkpoint (released separately on Zenodo/HuggingFace) and the cached embedding files, you can re-run downstream analyses without retraining:

```bash
# Frozen-encoder probes for GSD categories and individual diagnoses
python scripts/unified_gsd_probes_v3.py

# External-dataset transfer (5-fold CV with logistic regression)
python scripts/eval_h28_external_v3.py --dataset {neurovoz,coswara,svd,mdvr_kcl,mpower}

# NeuroVoz cross-lingual PD detection (5-seed mean ± SD)
python scripts/evaluate_neurovoz_v3.py

# eGeMAPSv02 acoustic decomposition (NeuroVoz; uses fold-internal scaling)
python scripts/gemaps_neurovoz_analysis.py

# Few-shot transfer
python scripts/eval_whisper_fewshot_v3.py

# Recording attribution (greedy forward selection across seeds)
python scripts/compute_recording_attribution_v3.py --seed 42  # repeat 42-46

# Embedding interpretability (acoustic grounding, NN retrieval, within-participant)
python scripts/evaluate_acoustic_grounding_v3.py
python scripts/evaluate_embedding_structure_v3.py
```

## Path C — Full pipeline from raw audio

You will need:
1. **Bridge2AI-Voice dataset** via PhysioNet (subject to data-use agreement; see `data/README.md`)
2. **External datasets:** mPower (Synapse syn4993293), NeuroVoz, MDVR-KCL, Saarbrücken Voice Database (SVD), Coswara
3. **A GPU node** — model training was run on NVIDIA A40/A100 GPUs; full 5-seed VoiceFM-Whisper training takes ~12 h per seed.

### 1. Preprocessing

```bash
# Generate participant + recording manifests with GSD labels
python scripts/preprocess.py

# Sanity-check expected counts (846 train + 138 validation, 40,056 recordings)
python -c "import pandas as pd; print(pd.read_parquet('data/processed/participants.parquet').shape)"
```

### 2. Training

5 seeds of VoiceFM-Whisper on the 846-participant training cohort (the released numbers used the per-seed configs `exp_whisper_ft4_gsd_seed{42,43,44,45,46}.yaml`):

```bash
for seed in 42 43 44 45 46; do
  python scripts/train.py --experiment exp_whisper_ft4_gsd_seed${seed}
done
```

For SLURM clusters, the wrapper template lives at:

```bash
sbatch scripts/h28_whisper_ft4_gsd_v3_slurm.sh   # cluster-specific; edit the SBATCH/module load block first
```

### 3. Embedding extraction

```bash
# 256d projection embeddings, mean-pooled per participant
python scripts/whisper_extract_embeddings_v3.py \
  --checkpoint <path/to/voicefm_whisper_seed42/best_model.pt> \
  --out results_v3/voicefm_whisper_embeddings.npz
```

### 4. Evaluation, figures

Same as Paths A and B above.

## SLURM templates

Every long-running job has a `*_slurm.sh` wrapper. **These are written for your HPC cluster and will not run unedited on other systems.** Before submitting, edit:

- `#SBATCH --partition=...` (your cluster's partition name)
- `module load mamba && source activate voicefm` (replace with your environment activation)
- `cd ~/VoiceFM` (replace with your project root)

A `# CLUSTER-SPECIFIC TEMPLATE` header at the top of each script flags the lines that need modifying.

## Issues / questions

Open a GitHub issue or contact the corresponding authors listed in the manuscript.
