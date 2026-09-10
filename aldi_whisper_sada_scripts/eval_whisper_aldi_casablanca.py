"""
Evaluate a Whisper-ALDi checkpoint on the Casablanca ALDi dataset stored locally.

Example:
    CUDA_VISIBLE_DEVICES=0 \\
    venv/bin/python scripts/eval_whisper_aldi_casablanca.py \\
      --checkpoint checkpoints/whisper_aldi_sada_30h_medium/checkpoint_epoch3.pt \\
      --dataset assets/casablanca/casablanca_audio_aldi \\
      --batch-size 8 --device cuda --output metrics_casablanca_30h_medium.json
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperFeatureExtractor
from datasets import load_from_disk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from train_whisper_aldi import WhisperALDiRegressor  # noqa: E402


class CasablancaAudioDataset(Dataset):
    def __init__(self, hf_path: str, split: str = "train"):
        ds = load_from_disk(hf_path)
        self.ds = ds[split] if isinstance(ds, dict) else ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        audio = row["audio"]
        wav = torch.tensor(audio["array"], dtype=torch.float32)
        sr = audio["sampling_rate"]
        return {
            "wav": wav,
            "sr": sr,
            "label": float(row["label"]),
            "id": row.get("id", idx),
        }


def collate_fn(batch: List[Dict[str, Any]], fe: WhisperFeatureExtractor, target_sr: int):
    feats_list = []
    attn_list = []
    labels = []
    ids = []
    for item in batch:
        wav = item["wav"]
        sr = item["sr"]
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        if sr != target_sr:
            wav = torch.tensor(
                torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
            )
        out = fe(wav.numpy(), sampling_rate=target_sr, return_tensors="pt")
        feats_list.append(out["input_features"][0])
        if "attention_mask" in out:
            attn_list.append(out["attention_mask"][0])
        else:
            attn_list.append(torch.ones(out["input_features"].shape[-1], dtype=torch.long))
        labels.append(item["label"])
        ids.append(item["id"])
    # pad features to max frames
    max_frames = max(feat.shape[-1] for feat in feats_list)
    padded_feats = []
    padded_attn = []
    for feat, attn in zip(feats_list, attn_list):
        pad = max_frames - feat.shape[-1]
        if pad > 0:
            feat = torch.nn.functional.pad(feat, (0, pad))
            attn = torch.nn.functional.pad(attn, (0, pad))
        padded_feats.append(feat)
        padded_attn.append(attn)
    feats = torch.stack(padded_feats, dim=0)
    attn = torch.stack(padded_attn, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return {"input_features": feats, "attention_mask": attn, "labels": labels, "ids": ids}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True, help="Path to load_from_disk() Casablanca dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--output", default=None, help="Optional path to write metrics JSON")
    ap.add_argument("--pred-csv", default=None, help="Optional path to write per-sample predictions CSV")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model_size = ckpt_args.get("model_size", "medium")
    freeze_encoder = ckpt_args.get("freeze_encoder", False)

    hf_kwargs = {"local_files_only": True} if args.local_only or os.environ.get("HF_HUB_OFFLINE") else {}
    fe = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{model_size}", **hf_kwargs)
    target_sr = fe.sampling_rate

    print(f"[info] loading dataset from {args.dataset} split={args.split}")
    dataset = CasablancaAudioDataset(args.dataset, split=args.split)
    print(f"[info] dataset rows: {len(dataset):,}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, fe, target_sr),
    )

    model = WhisperALDiRegressor(
        model_id=f"openai/whisper-{model_size}", freeze_encoder=freeze_encoder
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mse_loss = torch.nn.MSELoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    start = time.time()
    rows = []
    with torch.no_grad():
        for batch in loader:
            feats = batch["input_features"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            preds = model(feats, attention_mask=attn)
            loss = mse_loss(preds, labels)
            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            if len(all_preds) % 50 == 0:
                done = sum(len(x) for x in all_preds)
                print(f"[info] processed {done}/{len(dataset)} samples")
            # stash rows for optional CSV
            for i, pid in enumerate(batch["ids"]):
                rows.append(
                    {
                        "id": pid,
                        "predicted_aldi": float(preds[i].detach().cpu().item()),
                        "true_aldi": float(labels[i].detach().cpu().item()),
                    }
                )
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    mae = float(np.mean(np.abs(all_preds - all_labels)))
    from scipy.stats import pearsonr, spearmanr

    pearson = float(pearsonr(all_preds, all_labels)[0]) if len(all_preds) > 1 else float("nan")
    spearman = float(spearmanr(all_preds, all_labels)[0]) if len(all_preds) > 1 else float("nan")
    avg_loss = float(total_loss / max(1, len(loader)))
    metrics = {
        "test_loss": avg_loss,
        "test_mae": mae,
        "test_pearson": pearson,
        "test_spearman": spearman,
        "num_rows": len(dataset),
        "runtime_sec": round(time.time() - start, 1),
    }
    print(json.dumps(metrics, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[saved] {args.output}")

    if args.pred_csv:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(args.pred_csv, index=False)
        print(f"[saved] {args.pred_csv}")


if __name__ == "__main__":
    main()
