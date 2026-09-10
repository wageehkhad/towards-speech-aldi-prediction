"""
Minimal Whisper encoder + regression head for ALDi on SADA.
Workflow:
 1) python scripts/download_sada.py
 2) python scripts/score_sada_aldi.py
 3) python scripts/train_whisper_aldi.py --target-hours 1.0 --model-size medium
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import load_dataset, load_from_disk
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    get_cosine_schedule_with_warmup,
)

from submission_paths import HF_CACHE_DIR, MODELS_DIR, SADA_DIR, path_str

try:
    import wandb
except ImportError:
    wandb = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bin_score(x: float) -> str:
    if x < 0.33:
        return "low"
    if x < 0.66:
        return "mid"
    return "high"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fill_missing_durations(df: pd.DataFrame) -> pd.DataFrame:
    if "duration_sec" not in df.columns:
        df["duration_sec"] = np.nan

    needs_fill = df["duration_sec"].isna() | (df["duration_sec"] <= 0)
    if not needs_fill.any():
        return df
    print(f"[info] Filling {needs_fill.sum()} missing durations via torchaudio.info")
    for idx in tqdm(df[needs_fill].index, desc="duration_fix"):
        row = df.loc[idx]
        path = row.get("path", None)
        if isinstance(path, str) and os.path.isfile(path):
            try:
                info = torchaudio.info(path)
                sr = info.sample_rate
                df.at[idx, "duration_sec"] = info.num_frames / sr
            except Exception:
                df.at[idx, "duration_sec"] = np.nan
        else:
            df.at[idx, "duration_sec"] = np.nan
    return df


def select_balanced_subset(
    df: pd.DataFrame, target_hours: float, seed: int = 1337
) -> pd.DataFrame:
    df = df.copy()
    if "duration_sec" not in df.columns:
        df["duration_sec"] = np.nan
    if "ALDi_bin" not in df.columns:
        df["ALDi_bin"] = df["ALDi"].apply(bin_score)
    df = fill_missing_durations(df)
    df = df.dropna(subset=["duration_sec", "ALDi", "path"])
    df = df.sample(frac=1.0, random_state=seed)  # shuffle once

    # target_hours <= 0 means "use the full dataset"
    if target_hours <= 0:
        total_hours = df["duration_sec"].sum() / 3600
        print(
            f"[subset] using FULL dataset: {len(df)} rows | {total_hours:.2f} hrs | "
            f"bin counts: {df['ALDi_bin'].value_counts().to_dict()}"
        )
        return df.reset_index(drop=True)

    target_sec = target_hours * 3600
    per_bin_target = target_sec / 3
    selected_rows = []
    used_sec = 0.0

    for bin_name in ["low", "mid", "high"]:
        bin_df = df[df["ALDi_bin"] == bin_name]
        acc = 0.0
        for _, row in bin_df.iterrows():
            selected_rows.append(row)
            dur = float(row["duration_sec"])
            acc += dur
            used_sec += dur
            if acc >= per_bin_target or used_sec >= target_sec:
                break

    subset = pd.DataFrame(selected_rows).reset_index(drop=True)
    if "duration_sec" not in subset.columns:
        subset["duration_sec"] = np.nan
    total_hours = subset["duration_sec"].sum() / 3600
    print(
        f"[subset] {len(subset)} rows | {total_hours:.2f} hrs | bin counts: "
        f"{subset['ALDi_bin'].value_counts().to_dict()}"
    )
    return subset


def stratified_split(
    df: pd.DataFrame, eval_fraction: float, seed: int
) : 
    rng = np.random.default_rng(seed)
    train_parts = []
    eval_parts = []
    for bin_name, grp in df.groupby("ALDi_bin"):
        grp = grp.sample(frac=1.0, random_state=seed)
        n_eval = max(1, int(len(grp) * eval_fraction))
        eval_parts.append(grp.iloc[:n_eval])
        train_parts.append(grp.iloc[n_eval:])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    eval_df = pd.concat(eval_parts).reset_index(drop=True)
    print(
        f"[split] train={len(train_df)} eval={len(eval_df)} "
        f"| eval_fraction≈{eval_fraction}"
    )
    return train_df, eval_df


@dataclass
class AudioItem:
    input_features: torch.Tensor
    attention_mask: torch.Tensor
    label: torch.Tensor


class AudioALDiDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: WhisperFeatureExtractor,
        target_sr: int = 16000,
        dataset_cache_dir: str = path_str(HF_CACHE_DIR),
    ):
        self.df = df.reset_index(drop=True)
        self.fe = feature_extractor
        self.target_sr = target_sr
        self.dataset_cache_dir = dataset_cache_dir
        self._hf_ds = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> AudioItem:
        row = self.df.iloc[idx]
        path = row["path"]
        if isinstance(path, str) and os.path.isfile(path):
            wav, sr = torchaudio.load(path)
            if wav.dim() > 1:
                wav = wav.mean(dim=0, keepdim=True)
        else:
            split = row.get("split", "train")
            if split not in self._hf_ds:
                self._hf_ds[split] = load_dataset(
                    "MohamedRashad/SADA22", split=split, cache_dir=self.dataset_cache_dir
                )
            sample = self._hf_ds[split][int(row["id"])]
            audio = sample["audio"]
            wav = torch.tensor(audio["array"]).unsqueeze(0)
            sr = audio["sampling_rate"]

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        wav = wav.squeeze(0).numpy()
        feats = self.fe(
            wav,
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )
        input_feats = feats.input_features[0]  # (80, frames)
        mask = torch.ones(input_feats.shape[-1], dtype=torch.bool)
        label = torch.tensor(row["ALDi"], dtype=torch.float)
        return AudioItem(input_feats, mask, label)


def collate_batch(items: List[AudioItem]) -> dict:
    max_frames = max(item.input_features.shape[-1] for item in items)
    feats, masks, labels = [], [], []
    for item in items:
        pad = max_frames - item.input_features.shape[-1]
        if pad > 0:
            input_feat = torch.nn.functional.pad(item.input_features, (0, pad))
            mask = torch.nn.functional.pad(item.attention_mask, (0, pad))
        else:
            input_feat = item.input_features
            mask = item.attention_mask
        feats.append(input_feat)
        masks.append(mask.to(torch.long))
        labels.append(item.label)
    return {
        "input_features": torch.stack(feats),  # (B, 80, T)
        "attention_mask": torch.stack(masks),  # (B, T)
        "labels": torch.stack(labels),  # (B,)
    }


class WhisperALDiRegressor(nn.Module):
    def __init__(self, model_id: str, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(model_id).encoder
        hidden = self.encoder.config.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor):
        enc_outputs = self.encoder(
            input_features=input_features, attention_mask=attention_mask
        )
        enc_out = enc_outputs.last_hidden_state  # (B, T_enc, H)
        # Align attention mask to encoder time dimension via nearest-neighbor resize
        reduced_mask = torch.nn.functional.interpolate(
            attention_mask.unsqueeze(1).float(),
            size=enc_out.size(1),
            mode="nearest",
        ).squeeze(1)
        mask = reduced_mask.unsqueeze(-1)  # (B, T_enc, 1)
        enc_out = enc_out * mask
        summed = enc_out.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths
        pred = self.head(pooled).squeeze(-1)
        return pred


def evaluate(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    mse_loss = nn.MSELoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            feats = batch["input_features"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            preds = model(feats, attention_mask=mask)
            loss = mse_loss(preds, labels)
            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    mae = np.mean(np.abs(all_preds - all_labels))
    pearson = pearsonr(all_preds, all_labels)[0] if len(all_preds) > 1 else np.nan
    spearman = spearmanr(all_preds, all_labels)[0] if len(all_preds) > 1 else np.nan
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, mae, pearson, spearman


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb_project:
        if wandb is None:
            raise ImportError("wandb not installed; pip install wandb or omit --wandb-project")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )

    if args.hf_dataset_dir:
        train_ds = load_from_disk(os.path.join(args.hf_dataset_dir, "train"))
        df = train_ds.to_pandas()
        df = df.rename(columns={"audio": "path"})
        # extract actual file paths from audio dicts
        def _extract_path(x):
            if isinstance(x, dict) and "path" in x:
                return x["path"]
            return None
        df["path"] = df["path"].apply(_extract_path)
        df["split"] = "train"
    else:
        df = pd.read_csv(args.manifest)
    subset = select_balanced_subset(df, target_hours=args.target_hours, seed=args.seed)
    train_df, eval_df = stratified_split(subset, eval_fraction=args.eval_fraction, seed=args.seed)
    ensure_dir(args.output_dir)
    subset.to_csv(os.path.join(args.output_dir, "subset_all.csv"), index=False)
    train_df.to_csv(os.path.join(args.output_dir, "subset_train.csv"), index=False)
    eval_df.to_csv(os.path.join(args.output_dir, "subset_eval.csv"), index=False)

    fe = WhisperFeatureExtractor.from_pretrained(
        f"openai/whisper-{args.model_size}"
    )
    train_ds = AudioALDiDataset(
        train_df, fe, target_sr=fe.sampling_rate, dataset_cache_dir=args.dataset_cache_dir
    )
    eval_ds = AudioALDiDataset(
        eval_df, fe, target_sr=fe.sampling_rate, dataset_cache_dir=args.dataset_cache_dir
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_batch,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_batch,
    )
    test_loader = None
    if args.test_manifest:
        test_df = pd.read_csv(args.test_manifest)
        test_ds = AudioALDiDataset(
            test_df, fe, target_sr=fe.sampling_rate, dataset_cache_dir=args.dataset_cache_dir
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_batch,
        )

    model = WhisperALDiRegressor(
        model_id=f"openai/whisper-{args.model_size}",
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not args.no_dataparallel:
        print(f"[info] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Separate LRs for encoder vs head
    enc_params = [p for n, p in model.named_parameters() if "encoder" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "encoder" not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.lr_encoder},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=1e-4,
    )

    num_training_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = max(1, num_training_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    loss_fn = nn.MSELoss()

    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            feats = batch["input_features"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            preds = model(feats, attention_mask=mask)
            loss = loss_fn(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            global_step += 1

        avg_loss = running / max(1, len(train_loader))
        val_loss, val_mae, val_pearson, val_spearman = evaluate(model, eval_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_loss),
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
                "val_pearson": float(val_pearson) if val_pearson is not None else None,
                "val_spearman": float(val_spearman) if val_spearman is not None else None,
            }
        )
        if args.wandb_project:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "val_pearson": val_pearson,
                    "val_spearman": val_spearman,
                }
            )
        print(
            f"[epoch {epoch}] train_loss={avg_loss:.4f} | "
            f"val_loss={val_loss:.4f} mae={val_mae:.4f} "
            f"pearson={val_pearson:.3f} spearman={val_spearman:.3f}"
        )

        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(
            {
                "model_state": state,
                "args": vars(args),
                "epoch": epoch,
            },
            ckpt_path,
        )
        with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    test_metrics = None
    if test_loader is not None:
        test_loss, test_mae, test_pearson, test_spearman = evaluate(model, test_loader, device)
        test_metrics = {
            "test_loss": float(test_loss),
            "test_mae": float(test_mae),
            "test_pearson": float(test_pearson) if test_pearson is not None else None,
            "test_spearman": float(test_spearman) if test_spearman is not None else None,
        }
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(
            f"[test] loss={test_loss:.4f} mae={test_mae:.4f} "
            f"pearson={test_pearson:.3f} spearman={test_spearman:.3f}"
        )

    if args.wandb_project:
        if test_metrics is not None:
            wandb.log(test_metrics)
        wandb.finish()

    print("[done] Training completed")
    print(f"[info] final checkpoint → {ckpt_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=path_str(SADA_DIR / "manifest_train_aldi.csv"))
    ap.add_argument("--model-size", default="medium", choices=["small", "medium", "large"])
    ap.add_argument("--target-hours", type=float, default=1.0)
    ap.add_argument("--eval-fraction", type=float, default=0.1)
    ap.add_argument("--test-manifest", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--no-dataparallel", action="store_true", help="Force single-GPU even if multiple are visible")
    ap.add_argument("--output-dir", default=path_str(MODELS_DIR / "whisper_aldi_sada"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dataset-cache-dir", type=str, default=path_str(HF_CACHE_DIR))
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)
    ap.add_argument("--hf-dataset-dir", type=str, default=None, help="Path to load_from_disk() dataset (expects train/validation/test subdirs)")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
