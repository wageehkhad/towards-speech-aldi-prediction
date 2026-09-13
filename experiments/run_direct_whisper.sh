#!/usr/bin/env bash
# Direct setup, Whisper-medium encoder: audio -> ALDi score, no transcript.
#
#   stage 1  silver-standard labels from the SADA transcripts
#   stage 2  train the direct regression model (paper hyperparameters)
#   stage 3  evaluate on SADA-test, MediaSpeech and Casablanca
#   stage 4  cross-dataset error analysis
#
# Requires assets/ populated as described in assets/README.md.
# SKIP_TRAIN=1 reuses an existing checkpoint.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../aldi_whisper_sada_scripts"

CKPT_DIR=../assets/models/whisper_aldi_sada_full_medium

if [ ! -f ../assets/sada/manifest_train_aldi.csv ]; then
  echo "[stage] decompressing the training manifest"
  gunzip -k ../assets/sada/manifest_train_aldi.csv.gz
fi

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "[stage] scoring SADA transcripts with Sentence-ALDi"
  python score_sada_aldi.py

  echo "[stage] training the direct Whisper ALDi model"
  python train_whisper_aldi.py \
    --target-hours -1 --epochs 10 \
    --lr-encoder 5e-6 --lr-head 2e-4 \
    --output-dir "$CKPT_DIR"
fi

CKPT="$(python "$HERE/_best_checkpoint.py" "$CKPT_DIR")"
echo "[info] using best-validation checkpoint: $CKPT"

echo "[stage] evaluating on SADA-test"
python eval_whisper_aldi_test.py --checkpoint "$CKPT" \
  --test-manifest ../assets/sada/manifest_test_aldi.csv \
  --output ../results/direct_whisper_sada_test.json

echo "[stage] evaluating on MediaSpeech"
python eval_whisper_aldi_test.py --checkpoint "$CKPT" \
  --test-manifest ../assets/mediaspeech/manifest_mediaspeech_ar_aldi.csv \
  --output ../results/direct_whisper_mediaspeech.json

echo "[stage] evaluating on Casablanca"
python eval_whisper_aldi_casablanca.py --checkpoint "$CKPT" \
  --dataset ../assets/casablanca/casablanca_audio_aldi \
  --output ../results/direct_whisper_casablanca.json

echo "[stage] cross-dataset error analysis"
python error_analysis.py --checkpoint "$CKPT_DIR"

echo "[ok] direct Whisper setup complete; see analysis/ and results/"
