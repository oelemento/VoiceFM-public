# Data access

VoiceFM is trained and evaluated on datasets that are not redistributed in this repository. Each dataset has its own access process and data-use agreement (DUA). Once you have downloaded a dataset, place it under `data/raw/<dataset_name>/` and run `python scripts/preprocess.py` to produce the parquet manifests under `data/processed/` that the training and evaluation pipeline expects.

## Bridge2AI-Voice (B2AI Voice)

The primary training and prospective-validation dataset.

- **Source:** PhysioNet — https://doi.org/10.13026/k81f-qr68
- **Access:** Credentialed PhysioNet account + signed DUA
- **Citation:** Bensoussan Y, Sigaras A, Rameau A, et al. Bridge2AI-Voice: An ethically-sourced, diverse voice dataset linked to health information. *PhysioNet*. 2025.
- **Local layout (after download):**
  ```
  data/raw/b2ai-voice/
  ├── audio/<recording_id>.wav
  └── metadata/redcap_export.csv
  ```

## mPower (Parkinson's smartphone study)

Used for fine-tuned PD detection (Figure 6d–g).

- **Source:** Sage Bionetworks Synapse — https://www.synapse.org/Synapse:syn4993293
- **Access:** Synapse account + study DUA
- **Citation:** Bot BM, et al. The mPower study, Parkinson disease mobile data collected using ResearchKit. *Sci Data*. 2016;3:160011.
- **Note:** Audio files arrive as `.m4a`; convert to `.wav` with `scripts/convert_m4a_to_wav.py`.

## NeuroVoz (Spanish PD)

Used for cross-lingual transfer evaluation (Figure 6a–c).

- **Source:** https://github.com/BYO-UPM/NeuroVoz (or as cited in the original paper)
- **Citation:** Moro-Velázquez L, et al. *Biomed Signal Process Control*. 2019;48:205–220.

## MDVR-KCL (Parkinson's reading-passage)

Used for external transfer + few-shot evaluation (Figure 4).

- **Source:** Zenodo — https://doi.org/10.5281/zenodo.2867215
- **Citation:** Jaeger H, Trivedi D, Stadtschnitzer M. *Zenodo*. 2019.

## Saarbrücken Voice Database (SVD)

Used for external transfer evaluation (Figure 4).

- **Source:** https://stimmdb.coli.uni-saarland.de/
- **Citation:** Pützer M, Barry WJ. Saarbrücken Voice Database. Institute of Phonetics, Saarland University.

## Coswara (COVID-19 cough/breath)

Used as a non-voice acoustic transfer test (Figure 4).

- **Source:** https://github.com/iiscleap/Coswara-Data
- **Citation:** Sharma N, et al. *Proc. Interspeech 2020*. 2020:4811–4815.

## PVQD (held-out evaluation, supplementary)

Used in supplementary GRBAS / pathological-voice analyses.

- **Source:** Open Science Framework / dataset's home page; see citation
- **Note:** Optional — not required to reproduce the main figures.

## Trained model weights and cached embeddings

The pretrained VoiceFM-Whisper checkpoints (5 seeds × ~640 MB) and the cached 256-dimensional audio embeddings (used by Figures S2 and S3) are released separately under their own DOI. See the link in the published paper or open an issue if you need access.

## Expected directory layout after preprocessing

```
data/
├── raw/
│   ├── b2ai-voice/
│   ├── mpower/
│   ├── neurovoz/
│   ├── mdvr-kcl/
│   ├── svd/
│   └── coswara/
└── processed_v3/
    ├── participants.parquet     # 984 rows, GSD labels, train/test split
    ├── recordings.parquet       # 40,056 rows, with paths + durations
    └── capev_scores.parquet     # CAPE-V perceptual ratings
```
