#!/usr/bin/env python3
"""
Finetune Whisper ASR on SADA (audio -> transcript).

This script is intentionally verbose in runtime logging so stalls are easier to diagnose.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Audio, Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from submission_paths import MODELS_DIR, SADA_DIR, path_str

try:
    from jiwer import wer
except Exception:
    wer = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: str, manifest_dir: Optional[str] = None) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = [path, os.path.join(os.getcwd(), path)]
    if manifest_dir:
        candidates.append(os.path.join(manifest_dir, path))
    candidates.append(os.path.join(PROJECT_ROOT, path))
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return path


def normalize_manifest(df: pd.DataFrame, manifest_path: str) -> pd.DataFrame:
    out = df.copy()
    if "text" not in out.columns:
        for c in ["transcription", "transcript", "sentence"]:
            if c in out.columns:
                out = out.rename(columns={c: "text"})
                break
    if "path" not in out.columns:
        for c in ["audio_path", "audio"]:
            if c in out.columns:
                out = out.rename(columns={c: "path"})
                break
    if "text" not in out.columns or "path" not in out.columns:
        raise ValueError("Manifest missing required columns text/path: {}".format(manifest_path))
    if "duration_sec" not in out.columns:
        out["duration_sec"] = np.nan
    if "duration" in out.columns:
        miss = out["duration_sec"].isna()
        out.loc[miss, "duration_sec"] = out.loc[miss, "duration"]
    return out


def fill_missing_durations(df: pd.DataFrame, manifest_dir: str) -> pd.DataFrame:
    out = df.copy()
    miss = out["duration_sec"].isna() | (out["duration_sec"] <= 0)
    if not miss.any():
        return out
    print("[info] filling {} missing durations".format(int(miss.sum())))
    for idx in out[miss].index:
        path = resolve_path(str(out.at[idx, "path"]), manifest_dir=manifest_dir)
        if os.path.isfile(path):
            try:
                info = torchaudio.info(path)
                out.at[idx, "duration_sec"] = float(info.num_frames) / float(info.sample_rate)
            except Exception:
                out.at[idx, "duration_sec"] = np.nan
        else:
            out.at[idx, "duration_sec"] = np.nan
    return out


def prepare_manifest(manifest_path: str, max_samples: int = 0) -> pd.DataFrame:
    path = resolve_path(manifest_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    manifest_dir = os.path.dirname(path)
    df = pd.read_csv(path)
    df = normalize_manifest(df, path)
    df = fill_missing_durations(df, manifest_dir)

    resolved = []
    exists = []
    for p in df["path"].astype(str).tolist():
        rp = resolve_path(p, manifest_dir=manifest_dir)
        resolved.append(rp)
        exists.append(os.path.isfile(rp))
    df["path"] = resolved
    df["audio_exists"] = exists
    before = len(df)
    df = df[df["audio_exists"]].copy()
    if len(df) < before:
        print("[warn] dropped {} rows with missing audio".format(before - len(df)))

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    if max_samples and max_samples > 0:
        df = df.head(max_samples)
    df = df.reset_index(drop=True)
    print("[manifest] {} rows from {}".format(len(df), path))
    return df


# Draws a random subset up to a target number of audio hours, used for the
# training-volume sweep. Sampling is seeded and applied to the whole manifest
# before accumulating, so a given seed yields the same subset every run.
# target_hours <= 0 means use the full manifest.
def select_hours_subset(df: pd.DataFrame, target_hours: float, seed: int) -> pd.DataFrame:
    if target_hours <= 0:
        total_h = float(df["duration_sec"].fillna(0).sum()) / 3600.0
        print("[subset] full data rows={} hours={:.2f}".format(len(df), total_h))
        return df.reset_index(drop=True)

    target_sec = target_hours * 3600.0
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rows = []
    acc = 0.0
    for _, row in shuffled.iterrows():
        dur = float(row["duration_sec"]) if pd.notna(row["duration_sec"]) else 0.0
        if dur <= 0:
            continue
        rows.append(row)
        acc += dur
        if acc >= target_sec:
            break
    out = pd.DataFrame(rows).reset_index(drop=True)
    print("[subset] target_hours={} rows={} hours={:.2f}".format(target_hours, len(out), out["duration_sec"].sum() / 3600.0))
    return out


def build_dataset(df: pd.DataFrame) -> Dataset:
    keep = ["path", "text"]
    if "id" in df.columns:
        keep.append("id")
    if "duration_sec" in df.columns:
        keep.append("duration_sec")
    data = df[keep].rename(columns={"path": "audio"}).copy()
    ds = Dataset.from_pandas(data, preserve_index=False)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


@dataclass
class DataCollatorWhisperASR:
    processor: WhisperProcessor
    max_text_length: int
    decoder_start_token_id: int
    audio_max_length: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        audio_arrays = [f["audio"]["array"] for f in features]
        sampling_rate = int(features[0]["audio"]["sampling_rate"])

        feat_batch = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.audio_max_length,
            return_attention_mask=True,
        )
        text_batch = [str(f["text"]) for f in features]
        token_batch = self.processor.tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        )
        labels = token_batch["input_ids"].masked_fill(token_batch["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        return {"input_features": feat_batch["input_features"], "labels": labels}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", default=path_str(SADA_DIR / "manifest_train_aldi.csv"))
    ap.add_argument("--val-manifest", default=path_str(SADA_DIR / "manifest_val_aldi.csv"))
    ap.add_argument("--model-name", default="openai/whisper-medium")
    ap.add_argument("--language", default="arabic")
    ap.add_argument("--task", default="transcribe")
    ap.add_argument("--target-hours", type=float, default=30.0, help="<=0 means full train set")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--train-batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-text-length", type=int, default=256)
    ap.add_argument("--generation-max-length", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-eval-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--disable-fp16", action="store_true")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--output-dir", default=path_str(MODELS_DIR / "whisper_sada_asr"))
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    if args.local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("[info] loading manifests")
    train_df = prepare_manifest(args.train_manifest, max_samples=args.max_train_samples)
    val_df = prepare_manifest(args.val_manifest, max_samples=args.max_eval_samples)
    train_df = select_hours_subset(train_df, target_hours=args.target_hours, seed=args.seed)

    train_df.to_csv(os.path.join(args.output_dir, "train_subset.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "eval_subset.csv"), index=False)

    print("[info] loading processor/model {}".format(args.model_name))
    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task=args.task,
        local_files_only=bool(args.local_only or os.environ.get("HF_HUB_OFFLINE")),
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        local_files_only=bool(args.local_only or os.environ.get("HF_HUB_OFFLINE")),
    )
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language,
        task=args.task,
    )
    model.config.suppress_tokens = []
    model.config.use_cache = False

    print("[info] building datasets")
    train_ds = build_dataset(train_df)
    eval_ds = build_dataset(val_df)
    print("[info] train rows={} eval rows={}".format(len(train_ds), len(eval_ds)))

    collator = DataCollatorWhisperASR(
        processor=processor,
        max_text_length=args.max_text_length,
        decoder_start_token_id=model.config.decoder_start_token_id,
        audio_max_length=processor.feature_extractor.n_samples,
    )

    def compute_metrics(pred):
        if wer is None:
            return {"wer": float("nan")}
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_txt = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_txt = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_txt = [p.strip() for p in pred_txt]
        ref_txt = [r.strip() for r in ref_txt]
        try:
            return {"wer": float(wer(ref_txt, pred_txt))}
        except Exception:
            return {"wer": float("nan")}

    use_fp16 = bool(torch.cuda.is_available() and not args.disable_fp16)
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        dataloader_num_workers=args.num_workers,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        fp16=use_fp16,
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("[info] starting training")
    train_result = trainer.train()
    eval_metrics = trainer.evaluate(metric_key_prefix="eval")

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.output_dir, "train_result.json"), "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    print("[done] finetuning complete")
    print("[saved] {}".format(args.output_dir))
    print("[eval] {}".format(eval_metrics))


if __name__ == "__main__":
    main()
