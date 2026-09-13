#!/usr/bin/env bash
# Direct setup, MMS-1B encoder: audio -> ALDi score, no transcript.
#
# Same design as run_direct_whisper.sh with the MMS-1B encoder in place of
# Whisper. MMS-1B trains with FP16 and gradient accumulation for memory.
#
# Requires assets/ populated as described in assets/README.md.
# SKIP_TRAIN=1 reuses an existing checkpoint.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../aldi_whisper_sada_scripts"

CKPT_DIR=../assets/models/mms1b_aldi_sada

if [ ! -f ../assets/sada/manifest_train_aldi.csv ]; then
  echo "[stage] decompressing the training manifest"
  gunzip -k ../assets/sada/manifest_train_aldi.csv.gz
fi

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "[stage] training the direct MMS-1B ALDi model"
  python train_mms_aldi.py \
    --target-hours -1 --epochs 10 \
    --lr-encoder 2e-6 --lr-head 2e-4 \
    --output-dir "$CKPT_DIR"
fi

CKPT="$(python "$HERE/_best_checkpoint.py" "$CKPT_DIR")"
echo "[info] using best-validation checkpoint: $CKPT"

echo "[stage] evaluating on SADA-test"
python eval_mms_aldi.py --checkpoint "$CKPT" \
  --test-manifest ../assets/sada/manifest_test_aldi.csv \
  --output ../results/mms_eval_sada_test.json

echo "[stage] evaluating on MediaSpeech"
python eval_mms_aldi.py --checkpoint "$CKPT" \
  --test-manifest ../assets/mediaspeech/manifest_mediaspeech_ar_aldi.csv \
  --output ../results/mms_eval_mediaspeech.json

echo "[stage] evaluating on Casablanca"
python eval_mms_aldi.py --checkpoint "$CKPT" \
  --casablanca-hf-dataset ../assets/casablanca/casablanca_audio_aldi \
  --output ../results/mms_eval_casablanca.json

echo "[ok] direct MMS setup complete; see results/mms_eval_*.json"
