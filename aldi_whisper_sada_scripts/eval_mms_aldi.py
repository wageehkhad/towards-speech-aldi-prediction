"""
Evaluate MMS ALDi regressor checkpoints on:
1) CSV manifest (path + ALDi or aliases), e.g., MediaSpeech
2) Casablanca HF dataset saved via datasets.load_from_disk (audio + label), e.g., local Casablanca set
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import load_from_disk
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from train_mms_aldi import (
    AudioALDiDataset,
    AudioItem,
    MmsALDiRegressor,
    make_collate_fn,
)


# Manifests from different corpora name the target column differently; this
# renames whichever alias is present so the loader downstream sees one name.
def normalize_score_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ALDi" not in out.columns:
        for col in ["aldi_score", "label", "true_aldi", "aldi"]:
            if col in out.columns:
                out = out.rename(columns={col: "ALDi"})
                break
    return out


class CasablancaHFDataset(Dataset):
    def __init__(self, hf_dir: str, split: str, target_sr: int):
        ds_any = load_from_disk(hf_dir)
        self.ds = ds_any[split] if isinstance(ds_any, dict) else ds_any
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> AudioItem:
        row = self.ds[idx]
        audio = row["audio"]
        wav = torch.tensor(audio["array"], dtype=torch.float32)
        sr = int(audio["sampling_rate"])
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.target_sr).squeeze(0)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return AudioItem(wav=wav, label=label)


# Correlations return NaN rather than raising when a split is constant or too
# small, so a single degenerate subset does not abort the whole evaluation.
def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    try:
        return float(pearsonr(x, y)[0])
    except Exception:
        return float("nan")


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    try:
        return float(spearmanr(x, y)[0])
    except Exception:
        return float("nan")


def evaluate_with_predictions(model, loader, device, desc: str = "eval") -> Dict[str, object]:
    model.eval()
    mse_loss = torch.nn.MSELoss()
    total = 0.0
    all_preds, all_labels = [], []
    seen = 0
    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc=desc, unit="batch")
        for batch in pbar:
            vals = batch["input_values"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            preds = model(vals, attention_mask=mask)
            loss = mse_loss(preds, labels)
            total += float(loss.item())
            seen += int(labels.shape[0])
            all_preds.append(preds.detach().float().cpu().numpy())
            all_labels.append(labels.detach().float().cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "rows": seen})
        pbar.close()

    pred = np.concatenate(all_preds)
    true = np.concatenate(all_labels)
    err = pred - true
    out = {
        "n": int(len(pred)),
        "test_loss": float(total / max(1, len(loader))),
        "test_mae": float(np.mean(np.abs(err))),
        "test_mse": float(np.mean(err ** 2)),
        "test_pearson": safe_pearson(pred, true),
        "test_spearman": safe_spearman(pred, true),
        "error_mean": float(np.mean(err)),
        "error_std": float(np.std(err)),
        "predictions": pred.tolist(),
        "labels": true.tolist(),
    }
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint_epoch*.pt")
    ap.add_argument(
        "--test-manifest",
        default=None,
        help="CSV manifest with path + ALDi column (or aliases). For MediaSpeech-style eval.",
    )
    ap.add_argument(
        "--casablanca-hf-dataset",
        default=None,
        help="load_from_disk() Casablanca dataset path containing audio + label.",
    )
    ap.add_argument("--casablanca-split", default="train")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0, help="0 means full set")
    ap.add_argument("--output", required=True, help="Metrics JSON output path")
    ap.add_argument("--predictions-output", default=None, help="Optional CSV for predictions")
    return ap.parse_args()


def main():
    args = parse_args()
    if bool(args.test_manifest) == bool(args.casablanca_hf_dataset):
        raise ValueError("Pass exactly one of --test-manifest or --casablanca-hf-dataset.")

    if args.local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model_name = ckpt_args.get("model_name", "facebook/mms-1b-all")
    freeze_encoder = ckpt_args.get("freeze_encoder", False)
    unfreeze_last_n = ckpt_args.get("unfreeze_last_n", 0)

    hf_kwargs = {"local_files_only": True} if args.local_only or os.environ.get("HF_HUB_OFFLINE") else {}
    fe = AutoFeatureExtractor.from_pretrained(model_name, **hf_kwargs)
    target_sr = fe.sampling_rate or 16000
    collate_fn = make_collate_fn(fe)

    if args.test_manifest:
        df = pd.read_csv(args.test_manifest)
        df = normalize_score_col(df)
        if "ALDi" not in df.columns:
            raise ValueError(f"Missing ALDi (or alias) in {args.test_manifest}")
        if args.max_samples > 0:
            df = df.head(args.max_samples)
        ds = AudioALDiDataset(
            df,
            target_sr=target_sr,
            max_audio_sec=0.0,
            manifest_dir=os.path.dirname(os.path.abspath(args.test_manifest)),
        )
        dataset_name = "manifest_eval"
    else:
        ds = CasablancaHFDataset(args.casablanca_hf_dataset, split=args.casablanca_split, target_sr=target_sr)
        if args.max_samples > 0:
            class _SliceDataset(Dataset):
                def __init__(self, base, n):
                    self.base = base
                    self.n = min(len(base), n)
                def __len__(self):
                    return self.n
                def __getitem__(self, idx):
                    return self.base[idx]
            ds = _SliceDataset(ds, args.max_samples)
        dataset_name = f"casablanca_{args.casablanca_split}"

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = MmsALDiRegressor(
        model_name=model_name,
        freeze_encoder=freeze_encoder,
        unfreeze_last_n=unfreeze_last_n,
        local_files_only=bool(args.local_only or os.environ.get("HF_HUB_OFFLINE")),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    print(f"[info] evaluating {dataset_name} | rows={len(ds)} | batches={len(loader)}")
    out = evaluate_with_predictions(model, loader, device, desc=f"Eval [{dataset_name}]")
    out["dataset"] = dataset_name
    out["checkpoint"] = args.checkpoint
    out["model_name"] = model_name
    out["device"] = str(device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in out.items() if k not in {"predictions", "labels"}}, f, indent=2)
    print(f"[saved] {args.output}")
    print(
        f"[test] n={out['n']} loss={out['test_loss']:.4f} mae={out['test_mae']:.4f} "
        f"pearson={out['test_pearson']:.4f} spearman={out['test_spearman']:.4f}"
    )

    if args.predictions_output:
        pred_df = pd.DataFrame(
            {"true_aldi": out["labels"], "predicted_aldi": out["predictions"]}
        )
        os.makedirs(os.path.dirname(args.predictions_output), exist_ok=True)
        pred_df.to_csv(args.predictions_output, index=False)
        print(f"[saved] {args.predictions_output}")


if __name__ == "__main__":
    main()
