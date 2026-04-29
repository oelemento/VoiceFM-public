# VoiceFM — A Foundation Model for Clinical Voice Biomarkers

Code release accompanying the manuscript *"Towards A Foundation Model for Clinical Voice Biomarkers"* (Elemento et al., 2026).

VoiceFM is a CLIP-style dual-encoder model that learns general-purpose clinical voice representations by aligning audio recordings with rich clinical metadata via symmetric InfoNCE loss. The primary model (VoiceFM-Whisper) pairs a fine-tuned Whisper large-v2 audio encoder with a tabular transformer over 44 clinical features.

## Repository contents

```
VoiceFM-public/
├── src/                    model architecture, training, evaluation library
├── scripts/                training entry points, evaluation, figure generation
├── configs/                YAML configuration for model, data, training experiments
├── results_v3/             result JSONs from the published runs (used to regenerate figures)
├── data/                   access instructions for Bridge2AI-Voice and external datasets
├── REPRODUCING.md          step-by-step reproduction guide
├── LICENSE                 MIT
└── requirements.txt
```

This repository contains **code only**. Raw audio, trained model weights, and cached embeddings are distributed separately under their respective data-use agreements (see `data/README.md`).

## Quick start

```bash
git clone https://github.com/oelemento/VoiceFM-public.git
cd VoiceFM-public
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

To train a VoiceFM-Whisper model from scratch you will need access to the Bridge2AI-Voice dataset (see `data/README.md`). Once data are available:

```bash
# 1. Build participant + recording manifests
python scripts/preprocess.py

# 2. Train (single seed, single GPU; SLURM templates in scripts/*_slurm.sh for clusters)
python scripts/train.py --experiment exp_whisper_ft4_gsd_seed42

# 3. Extract frozen 256d audio embeddings for the cohort
python scripts/whisper_extract_embeddings_v3.py \
  --checkpoint <path/to/best_model.pt> \
  --out results_v3/voicefm_whisper_recording_embeddings.npz

# 4. Run gold-standard-diagnosis probes
python scripts/unified_gsd_probes_v3.py
```

For the full pipeline including external datasets, prospective validation, and figure regeneration, see `REPRODUCING.md`.

## Reproducing the paper figures

If you have only the released result JSONs in `results_v3/`, you can regenerate every figure in the paper without retraining:

```bash
python scripts/paper_fig2a_results_v3.py
python scripts/paper_fig2b_diagnoses_v3.py
python scripts/paper_fig3_prospective_v3.py
python scripts/paper_fig5_composite_v3.py
# ...one script per figure, or assemble them all into one PDF:
python scripts/paper_assemble_pdf_v3.py
```

A few figures (S2 acoustic grounding, S3 embedding structure, t-SNE) require the cached embedding files (`*.npz`); these are released separately. See `REPRODUCING.md` for download instructions.

## Citation

If you use VoiceFM, please cite:

```
@article{elemento2026voicefm,
  title   = {Towards A Foundation Model for Clinical Voice Biomarkers},
  author  = {Elemento, Olivier and Sigaras, Alexandros and Colonel, Joseph T.
             and Ghosh, Satrajit S. and Bensoussan, Yael and
             Bridge2AI-Voice Consortium and Rameau, Ana{\"i}s},
  year    = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The Bridge2AI-Voice dataset and other external datasets remain subject to their original data-use agreements.

## Acknowledgements

This work was supported by the NIH Bridge2AI Common Fund Program (OT2OD032720, Bridge2AI-Voice) and NIH R01DC020135. We thank the Bridge2AI-Voice Consortium and all study participants.
