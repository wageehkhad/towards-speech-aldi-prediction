# Towards Speech ALDi

Code and results for *Towards Speech ALDi: Predicting the Arabic Level of
Dialectness Directly from Speech* (ArabicNLP 2026).

ALDi scores Arabic text on a continuous [0, 1] scale from MSA to highly
dialectal. This repo extends it to speech, comparing a cascaded approach (ASR,
then text ALDi on the transcript) against direct regression from audio. Models
are fine-tuned on SADA and tested on SADA-test, MediaSpeech, and Casablanca.
Targets are Sentence-ALDi scores of the manual transcripts.

## Layout

```text
aldi_whisper_sada_scripts/              data prep, training, evaluation, baselines
genre_based_aldi_experiments_scripts/   genre validation study
assets/                                 manifests and metadata
analysis/                               error analysis, text vs speech comparison
results/                                metrics, comparison tables, statistical tests
```

## Script guide

### `aldi_whisper_sada_scripts/`

Data preparation

| Script | Purpose |
|---|---|
| `download_sada.py` | Downloads or prepares SADA audio and writes the CSV manifests |
| `score_sada_aldi.py` | Runs the text Sentence-ALDi scorer over transcripts to create the supervision labels |
| `build_sada_aldi_hf.py` | Converts the labelled manifests into a saved Hugging Face dataset layout |
| `submission_paths.py` | Centralises the repository-relative paths every script resolves through |

Training

| Script | Purpose |
|---|---|
| `train_whisper_aldi.py` | Direct Whisper-based speech-to-ALDi regression model |
| `train_mms_aldi.py` | Direct MMS-based speech-to-ALDi regression model |
| `finetune_whisper_asr.py` | Fine-tunes Whisper ASR on SADA for the cascaded baseline |

Evaluation

| Script | Purpose |
|---|---|
| `eval_whisper_aldi_test.py` | Evaluates a Whisper ALDi checkpoint on a CSV manifest such as SADA-test |
| `eval_whisper_aldi_casablanca.py` | Evaluates a Whisper ALDi checkpoint on the Casablanca dataset |
| `eval_mms_aldi.py` | Evaluates an MMS ALDi checkpoint on a manifest or the Casablanca dataset |
| `predict_whisper_aldi.py` | Single- or multi-file direct ALDi prediction |

Cascaded baselines

| Script | Purpose |
|---|---|
| `asr_aldi_baseline_utils.py` | Shared utilities for the ASR-to-Sentence-ALDi baselines |
| `baseline1_pretrained_asr.py` | Zero-shot Whisper ASR followed by Sentence-ALDi |
| `baseline2_finetuned_asr.py` | SADA-finetuned Whisper ASR followed by Sentence-ALDi |

Analysis and reporting

| Script | Purpose | Produces |
|---|---|---|
| `compare_baselines.py` | Aggregates the direct-vs-baseline comparison | Table 1 |
| `analyze_b2_vs_direct_wer.py` | Relates WER to ALDi error, per dialect and in WER bins | Table 2 RMSE and WER columns |
| `dialect_ttests.py` | Per-dialect paired t-tests on squared error | Table 2 significance markers |
| `plot_scaling.py` | RMSE against training hours | Appendix scaling figure |
| `error_analysis.py` | Cross-dataset error analysis for the direct model | |
| `error_analysis_filtered.py` | Error analysis variant with optional non-speech filtering | |
| `text_vs_speech_aldi.py` | Direct speech predictions against text Sentence-ALDi scores | |

### `genre_based_aldi_experiments_scripts/`

Run in this order; each step feeds the next.

| Script | Purpose |
|---|---|
| `download_youtube_audio.py` | Downloads the trimmed audio window for each source in `sources.csv` |
| `segment_audio.py` | Cuts each trimmed file into fixed 15-second clips |
| `score_genre_clips.py` | Scores every clip with a direct Whisper ALDi checkpoint |
| `summarize_genre_predictions.py` | Aggregates into per-genre and per-source summaries and the box plot |

Note that the scripts in each directory import each other as siblings, so keep
them where they are and run them from inside their own directory.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Python 3.8, CUDA PyTorch. Trained on NVIDIA Quadro RTX 8000.

## Data

Manifests and metadata only. No audio, no checkpoints. Get SADA, Casablanca and
MediaSpeech from their original sources and lay them out under `assets/` as
described in `assets/README.md`.

The SADA training manifest is gzipped:

```bash
gunzip -k assets/sada/manifest_train_aldi.csv.gz
```

## Running the experiments

All paths below are relative to `aldi_whisper_sada_scripts/`.

### Labels

```bash
python download_sada.py
python score_sada_aldi.py
python build_sada_aldi_hf.py            # optional, faster loading
```

### Direct models

```bash
python train_whisper_aldi.py \
  --target-hours -1 --epochs 10 \
  --lr-encoder 5e-6 --lr-head 2e-4 \
  --output-dir ../assets/models/whisper_aldi_sada_full_medium

python train_mms_aldi.py \
  --target-hours -1 --epochs 10 \
  --lr-encoder 2e-6 --lr-head 2e-4 \
  --output-dir ../assets/models/mms1b_aldi_sada
```

AdamW, cosine schedule with 10% linear warmup, gradient clipping at 1.0, MSE
loss. MMS-1B uses FP16 and gradient accumulation. Best validation checkpoint is
kept.

```bash
python eval_whisper_aldi_test.py \
  --checkpoint ../assets/models/whisper_aldi_sada_full_medium/checkpoint_epoch5.pt \
  --test-manifest ../assets/sada/manifest_test_aldi.csv

python eval_whisper_aldi_casablanca.py \
  --checkpoint ../assets/models/whisper_aldi_sada_full_medium/checkpoint_epoch5.pt \
  --dataset ../assets/casablanca/casablanca_audio_aldi

python eval_mms_aldi.py \
  --checkpoint ../assets/models/mms1b_aldi_sada/checkpoint_epoch5.pt \
  --test-manifest ../assets/mediaspeech/manifest_mediaspeech_ar_aldi.csv \
  --output ../results/mms_eval.json
```

### Cascaded baselines

```bash
python finetune_whisper_asr.py          # SADA-finetuned ASR
python baseline1_pretrained_asr.py      # zero-shot Whisper + Sentence-ALDi
python baseline2_finetuned_asr.py       # finetuned Whisper + Sentence-ALDi
```

### Comparison and analysis

```bash
python compare_baselines.py             # Table 1, paired tests, per-dialect tables
python analyze_b2_vs_direct_wer.py      # WER column of Table 2, WER bins
python error_analysis.py
python text_vs_speech_aldi.py
```

### Scaling

Train at several values of `--target-hours`, record one row per run in
`results/experiments.csv` (`run_id, model, train_hours, test_set, rmse, ...`),
then draw the figure:

```bash
python plot_scaling.py
```

Writes `results/scaling_figure_rmse.pdf` and `.png`. Any test set present in the
file gets its own panel, so a partly filled file still plots.

### Genre study

15-second clips from 49 YouTube sources across six genres, to check the direct
model tracks register rather than accent alone.

```bash
cd ../genre_based_aldi_experiments_scripts
python download_youtube_audio.py
python segment_audio.py
python score_genre_clips.py --checkpoint ../assets/models/whisper_aldi_sada_full_medium/checkpoint_epoch5.pt
python summarize_genre_predictions.py
```

Source URLs, genres, dialect regions and time windows are in
`assets/genre_based_aldi/sources.csv`. Three videos have since been taken down;
their URLs are kept for the record.

## Results

- `results/baseline_comparison/` RMSE tables, per-dialect Casablanca results, paired t-tests
- `results/baseline_comparison/wer_analysis/` per-dialect WER, WER-binned errors
- `results/genre_based_aldi/` per-genre and per-source summaries
- `results/baseline1_pretrained_asr/`, `results/baseline2_finetuned_asr/` baseline metrics
- `analysis/error_analysis_full_sada/` errors by dialect, duration and ALDi range
- `analysis/text_vs_speech_aldi/` direct predictions against text Sentence-ALDi

Per-utterance prediction dumps are not included; rerun the eval scripts above.

## Models

- Direct speech ALDi: https://huggingface.co/wageehkhad/whisper-medium-sada-speech-aldi
- SADA-finetuned ASR baseline: https://huggingface.co/wageehkhad/whisper-medium-finetuned-sada-asr

Text scorer used for supervision: https://huggingface.co/AMR-KELEG/Sentence-ALDi
