#!/usr/bin/env python3
"""
Baseline 2:
Audio -> SADA-finetuned Whisper ASR -> transcript -> Sentence-ALDi score.
"""

import argparse
import os
from typing import Dict

from asr_aldi_baseline_utils import (
    SentenceALDiScorer,
    WhisperASRTranscriber,
    iter_casablanca_hf_samples,
    load_manifest_samples,
    run_asr_sentence_aldi,
    save_metrics_summary,
)
from submission_paths import CASABLANCA_DIR, MEDIASPEECH_DIR, MODELS_DIR, RESULTS_DIR, SADA_DIR, path_str


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--asr-model",
        default=path_str(MODELS_DIR / "whisper_sada_asr"),
        help="Path to finetuned Whisper ASR checkpoint directory.",
    )
    ap.add_argument("--aldi-model", default="AMR-KELEG/Sentence-ALDi")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--clamp-aldi-output", action="store_true")

    ap.add_argument(
        "--sada-manifest",
        default=path_str(SADA_DIR / "manifest_test_aldi.csv"),
    )
    ap.add_argument(
        "--casablanca-manifest",
        default=path_str(CASABLANCA_DIR / "casablanca_aldi_scored.csv"),
    )
    ap.add_argument(
        "--casablanca-hf-dataset",
        default=path_str(CASABLANCA_DIR / "casablanca_audio_aldi"),
    )
    ap.add_argument("--casablanca-split", default="train")
    ap.add_argument(
        "--mediaspeech-manifest",
        default=path_str(MEDIASPEECH_DIR / "manifest_mediaspeech_ar_aldi.csv"),
    )

    ap.add_argument("--asr-batch-size", type=int, default=8)
    ap.add_argument("--aldi-batch-size", type=int, default=64)
    ap.add_argument("--max-text-length", type=int, default=256)
    ap.add_argument("--chunk-length-s", type=float, default=30.0)
    ap.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=0,
        help="0 means full dataset.",
    )
    ap.add_argument(
        "--output-dir",
        default=path_str(RESULTS_DIR / "baseline2_finetuned_asr"),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.asr_model):
        raise FileNotFoundError(
            "ASR model path not found: {}. Run finetune_whisper_asr.py first.".format(args.asr_model)
        )

    asr = WhisperASRTranscriber(
        asr_model=args.asr_model,
        device=args.device,
        chunk_length_s=args.chunk_length_s,
        local_only=args.local_only,
    )
    scorer = SentenceALDiScorer(
        model_id=args.aldi_model,
        device=args.device,
        local_only=args.local_only,
        clamp_output=args.clamp_aldi_output,
    )

    max_n = args.max_samples_per_dataset if args.max_samples_per_dataset > 0 else 0
    summary: Dict[str, Dict[str, float]] = {}

    # 1) SADA test
    sada_samples = load_manifest_samples(
        manifest_csv=args.sada_manifest,
        dataset_name="sada_test",
        max_samples=max_n,
    )
    _, metrics_sada = run_asr_sentence_aldi(
        samples=sada_samples,
        asr=asr,
        scorer=scorer,
        output_csv=os.path.join(args.output_dir, "predictions_sada_test.csv"),
        asr_batch_size=args.asr_batch_size,
        aldi_batch_size=args.aldi_batch_size,
        max_length=args.max_text_length,
        progress_desc="Baseline2 [sada_test]",
    )
    summary["sada_test"] = metrics_sada
    print("[sada_test] {}".format(metrics_sada))

    # 2) Casablanca
    casa_samples = iter_casablanca_hf_samples(
        hf_dataset_dir=args.casablanca_hf_dataset,
        split=args.casablanca_split,
        metadata_csv=args.casablanca_manifest,
        max_samples=max_n,
    )
    _, metrics_casa = run_asr_sentence_aldi(
        samples=casa_samples,
        asr=asr,
        scorer=scorer,
        output_csv=os.path.join(args.output_dir, "predictions_casablanca.csv"),
        asr_batch_size=args.asr_batch_size,
        aldi_batch_size=args.aldi_batch_size,
        max_length=args.max_text_length,
        progress_desc="Baseline2 [casablanca]",
    )
    summary["casablanca"] = metrics_casa
    print("[casablanca] {}".format(metrics_casa))

    # 3) MediaSpeech
    med_samples = load_manifest_samples(
        manifest_csv=args.mediaspeech_manifest,
        dataset_name="mediaspeech",
        max_samples=max_n,
    )
    _, metrics_med = run_asr_sentence_aldi(
        samples=med_samples,
        asr=asr,
        scorer=scorer,
        output_csv=os.path.join(args.output_dir, "predictions_mediaspeech.csv"),
        asr_batch_size=args.asr_batch_size,
        aldi_batch_size=args.aldi_batch_size,
        max_length=args.max_text_length,
        progress_desc="Baseline2 [mediaspeech]",
    )
    summary["mediaspeech"] = metrics_med
    print("[mediaspeech] {}".format(metrics_med))

    save_metrics_summary(
        path=os.path.join(args.output_dir, "metrics_summary.json"),
        metrics_by_dataset=summary,
    )
    print("[done] baseline2 outputs -> {}".format(args.output_dir))


if __name__ == "__main__":
    main()
