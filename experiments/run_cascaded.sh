#!/usr/bin/env bash
# Cascaded setup: ASR transcription followed by text ALDi estimation.
#
#   stage 1  fine-tune Whisper-medium for ASR on the full SADA training split
#   stage 2  zero-shot Whisper-medium -> Sentence-ALDi     (Table 1, Zero-shot)
#   stage 3  fine-tuned Whisper -> Sentence-ALDi           (Table 1, Fine-tuned)
#   stage 4  comparison tables, WER analysis, per-dialect t-tests
#
# Requires assets/ populated as described in assets/README.md.
# SKIP_TRAIN=1 reuses an existing ASR checkpoint.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../aldi_whisper_sada_scripts"

ASR_DIR=../assets/models/whisper_sada_asr_full

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "[stage] fine-tuning Whisper ASR on SADA (full training split)"
  python finetune_whisper_asr.py --target-hours -1 --output-dir "$ASR_DIR"
fi

echo "[stage] baseline 1: zero-shot Whisper -> Sentence-ALDi"
python baseline1_pretrained_asr.py

echo "[stage] baseline 2: fine-tuned Whisper -> Sentence-ALDi"
python baseline2_finetuned_asr.py --asr-model "$ASR_DIR"

echo "[stage] comparison, WER analysis and per-dialect t-tests"
python compare_baselines.py
python analyze_b2_vs_direct_wer.py
python dialect_ttests.py

echo "[ok] cascaded setup complete; see results/baseline_comparison/"
