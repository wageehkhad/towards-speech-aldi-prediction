"""
Train MMS encoder + regression head for ALDi prediction on SADA manifests.

Design mirrors train_whisper_aldi.py:
- Balanced subset selection by ALDi_bin for target hours (1/30/150/full)
- Stratified train/val split
- Per-epoch checkpointing + train_history.json
- Optional test-manifest final metrics to test_metrics.json
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    get_cosine_schedule_with_warmup,
)

from submission_paths import MODELS_DIR, SADA_DIR, path_str

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
        path = df.at[idx, "path"] if "path" in df.columns else None
        if isinstance(path, str) and os.path.isfile(path):
            try:
                info = torchaudio.info(path)
                df.at[idx, "duration_sec"] = info.num_frames / info.sample_rate
            except Exception:
                df.at[idx, "duration_sec"] = np.nan
        else:
            df.at[idx, "duration_sec"] = np.nan
    return df


def select_balanced_subset(df: pd.DataFrame, target_hours: float, seed: int) -> pd.DataFrame:
    df = df.copy()
    if "duration_sec" not in df.columns:
        df["duration_sec"] = np.nan
    if "ALDi_bin" not in df.columns:
        df["ALDi_bin"] = df["ALDi"].apply(bin_score)
    df = fill_missing_durations(df)
    df = df.dropna(subset=["duration_sec", "ALDi", "path"])
    df = df.sample(frac=1.0, random_state=seed)

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
    total_hours = subset["duration_sec"].sum() / 3600 if len(subset) else 0.0
    print(
        f"[subset] {len(subset)} rows | {total_hours:.2f} hrs | "
        f"bin counts: {subset['ALDi_bin'].value_counts().to_dict() if len(subset) else {}}"
    )
    return subset


def stratified_split(df: pd.DataFrame, eval_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    eval_parts = []
    for _, grp in df.groupby("ALDi_bin"):
        grp = grp.sample(frac=1.0, random_state=seed)
        n_eval = max(1, int(len(grp) * eval_fraction))
        eval_parts.append(grp.iloc[:n_eval])
        train_parts.append(grp.iloc[n_eval:])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    eval_df = pd.concat(eval_parts).reset_index(drop=True)
    print(f"[split] train={len(train_df)} eval={len(eval_df)} | eval_fraction≈{eval_fraction}")
    return train_df, eval_df


def resolve_audio_path(path: str, manifest_dir: Optional[str]) -> str:
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    candidates = [
        path,
        os.path.join(os.getcwd(), path),
    ]
    if manifest_dir:
        candidates.append(os.path.join(manifest_dir, path))
    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return path


@dataclass
class AudioItem:
    wav: torch.Tensor
    label: torch.Tensor


class AudioALDiDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_sr: int,
        max_audio_sec: float = 0.0,
        manifest_dir: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.target_sr = target_sr
        self.max_audio_sec = max_audio_sec
        self.max_samples = int(target_sr * max_audio_sec) if max_audio_sec and max_audio_sec > 0 else 0
        self.manifest_dir = manifest_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> AudioItem:
        row = self.df.iloc[idx]
        path = resolve_audio_path(str(row["path"]), self.manifest_dir)
        wav, sr = torchaudio.load(path)
        if wav.dim() > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        wav = wav.squeeze(0)
        # Optional truncation to protect memory on rare long clips.
        if self.max_samples > 0 and wav.numel() > self.max_samples:
            wav = wav[: self.max_samples]
        label = torch.tensor(float(row["ALDi"]), dtype=torch.float32)
        return AudioItem(wav=wav, label=label)


def make_collate_fn(feature_extractor: AutoFeatureExtractor):
    target_sr = feature_extractor.sampling_rate or 16000

    def collate(items: List[AudioItem]) -> Dict[str, torch.Tensor]:
        wavs = [it.wav.numpy() for it in items]
        labels = torch.stack([it.label for it in items])
        feats = feature_extractor(
            wavs,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        # Always create attention_mask if extractor omitted it.
        if "attention_mask" not in feats:
            feats["attention_mask"] = torch.ones_like(feats["input_values"], dtype=torch.long)
        return {
            "input_values": feats["input_values"],
            "attention_mask": feats["attention_mask"].to(torch.long),
            "labels": labels,
        }

    return collate


class MmsALDiRegressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze_encoder: bool = False,
        unfreeze_last_n: int = 0,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        hidden = getattr(self.encoder.config, "hidden_size", None)
        if hidden is None:
            raise ValueError("Could not infer hidden_size from encoder config.")
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if unfreeze_last_n > 0:
                layers = None
                # Wav2Vec2-like encoder blocks
                if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layers"):
                    layers = self.encoder.encoder.layers
                # Some models nest under .wav2vec2.encoder.layers
                elif hasattr(self.encoder, "wav2vec2") and hasattr(self.encoder.wav2vec2, "encoder"):
                    if hasattr(self.encoder.wav2vec2.encoder, "layers"):
                        layers = self.encoder.wav2vec2.encoder.layers
                if layers is not None:
                    for layer in layers[-unfreeze_last_n:]:
                        for p in layer.parameters():
                            p.requires_grad = True

    def _reduce_attention_mask(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Best path for wav2vec2-family models.
        if hasattr(self.encoder, "_get_feature_vector_attention_mask"):
            reduced = self.encoder._get_feature_vector_attention_mask(  # type: ignore[attr-defined]
                hidden_states.shape[1],
                attention_mask,
            )
            return reduced
        # Fallback for generic audio encoders.
        reduced = torch.nn.functional.interpolate(
            attention_mask.unsqueeze(1).float(),
            size=hidden_states.size(1),
            mode="nearest",
        ).squeeze(1)
        return reduced.to(attention_mask.dtype)

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, H)
        reduced_mask = self._reduce_attention_mask(hidden, attention_mask).unsqueeze(-1).float()
        hidden = hidden * reduced_mask
        pooled = hidden.sum(dim=1) / reduced_mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float, float]:
    model.eval()
    mse_loss = nn.MSELoss()
    total = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            vals = batch["input_values"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            preds = model(vals, attention_mask=mask)
            loss = mse_loss(preds, labels)
            total += loss.item()
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    mae = float(np.mean(np.abs(preds - labels)))
    pearson = float(pearsonr(preds, labels)[0]) if len(preds) > 1 else float("nan")
    spearman = float(spearmanr(preds, labels)[0]) if len(preds) > 1 else float("nan")
    avg_loss = float(total / max(1, len(loader)))
    return avg_loss, mae, pearson, spearman


def train(args):
    set_seed(args.seed)
    if args.local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[info] device={device}")

    if args.wandb_project:
        if wandb is None:
            raise ImportError("wandb not installed; install it or omit --wandb-project")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )

    manifest_path = args.manifest
    train_df = pd.read_csv(manifest_path)
    if "ALDi" not in train_df.columns:
        raise ValueError(f"Manifest {manifest_path} must include ALDi column.")
    subset = select_balanced_subset(train_df, target_hours=args.target_hours, seed=args.seed)
    tr_df, va_df = stratified_split(subset, eval_fraction=args.eval_fraction, seed=args.seed)

    ensure_dir(args.output_dir)
    subset.to_csv(os.path.join(args.output_dir, "subset_all.csv"), index=False)
    tr_df.to_csv(os.path.join(args.output_dir, "subset_train.csv"), index=False)
    va_df.to_csv(os.path.join(args.output_dir, "subset_eval.csv"), index=False)

    fe = AutoFeatureExtractor.from_pretrained(
        args.model_name,
        local_files_only=bool(args.local_only or os.environ.get("HF_HUB_OFFLINE")),
    )
    target_sr = fe.sampling_rate or 16000
    collate_fn = make_collate_fn(fe)
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))

    tr_ds = AudioALDiDataset(
        tr_df,
        target_sr=target_sr,
        max_audio_sec=args.max_audio_sec,
        manifest_dir=manifest_dir,
    )
    va_ds = AudioALDiDataset(
        va_df,
        target_sr=target_sr,
        max_audio_sec=args.max_audio_sec,
        manifest_dir=manifest_dir,
    )
    tr_loader = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    te_loader = None
    if args.test_manifest:
        test_df = pd.read_csv(args.test_manifest)
        te_ds = AudioALDiDataset(
            test_df,
            target_sr=target_sr,
            max_audio_sec=args.max_audio_sec,
            manifest_dir=os.path.dirname(os.path.abspath(args.test_manifest)),
        )
        te_loader = DataLoader(
            te_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    model = MmsALDiRegressor(
        model_name=args.model_name,
        freeze_encoder=args.freeze_encoder,
        unfreeze_last_n=args.unfreeze_last_n,
        local_files_only=bool(args.local_only or os.environ.get("HF_HUB_OFFLINE")),
    ).to(device)

    if args.grad_checkpointing:
        if hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()
            print("[info] Enabled gradient checkpointing on encoder")
        else:
            print("[warn] Encoder does not expose gradient_checkpointing_enable(); skipping")
        if hasattr(model.encoder, "config") and hasattr(model.encoder.config, "use_cache"):
            model.encoder.config.use_cache = False

    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not args.no_dataparallel:
        print(f"[info] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    enc_params = [p for n, p in model.named_parameters() if "encoder" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.lr_encoder},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, int(np.ceil(len(tr_loader) / max(1, args.grad_accum))))
    num_training_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, int(num_training_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    loss_fn = nn.MSELoss()
    use_amp = (device.type == "cuda") and (not args.disable_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    oom_batches = 0

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        pbar = tqdm(enumerate(tr_loader, start=1), total=len(tr_loader), desc=f"epoch {epoch}")
        for step, batch in pbar:
            vals = batch["input_values"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            try:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    preds = model(vals, attention_mask=mask)
                    loss = loss_fn(preds, labels)
                    loss_scaled = loss / max(1, args.grad_accum)

                scaler.scale(loss_scaled).backward()
                running += float(loss.item())

                should_step = (step % max(1, args.grad_accum) == 0) or (step == len(tr_loader))
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "oom": oom_batches})
            except RuntimeError as e:
                is_oom = "out of memory" in str(e).lower()
                if not (args.skip_oom_batches and is_oom):
                    raise
                oom_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pbar.set_postfix({"loss": "oom-skip", "oom": oom_batches})
                continue

        avg_train_loss = running / max(1, len(tr_loader))
        val_loss, val_mae, val_pearson, val_spearman = evaluate(model, va_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "val_loss": float(val_loss),
            "val_mae": float(val_mae),
            "val_pearson": float(val_pearson),
            "val_spearman": float(val_spearman),
            "oom_batches_total": int(oom_batches),
        }
        history.append(row)
        print(
            f"[epoch {epoch}] train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} mae={val_mae:.4f} "
            f"pearson={val_pearson:.3f} spearman={val_spearman:.3f}"
        )

        if args.wandb_project:
            wandb.log(row)

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
    if te_loader is not None:
        test_loss, test_mae, test_pearson, test_spearman = evaluate(model, te_loader, device)
        test_metrics = {
            "test_loss": float(test_loss),
            "test_mae": float(test_mae),
            "test_pearson": float(test_pearson),
            "test_spearman": float(test_spearman),
        }
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(
            f"[test] loss={test_loss:.4f} mae={test_mae:.4f} "
            f"pearson={test_pearson:.3f} spearman={test_spearman:.3f}"
        )
        if args.wandb_project:
            wandb.log(test_metrics)

    if args.wandb_project:
        wandb.finish()

    print("[done] Training completed")
    print(f"[info] final checkpoint → {ckpt_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=path_str(SADA_DIR / "manifest_train_aldi.csv"))
    ap.add_argument("--test-manifest", default=path_str(SADA_DIR / "manifest_test_aldi.csv"))
    ap.add_argument("--model-name", default="facebook/mms-1b-all")
    ap.add_argument("--target-hours", type=float, default=1.0, help="<=0 uses full dataset")
    ap.add_argument("--eval-fraction", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr-encoder", type=float, default=2e-6)
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--unfreeze-last-n", type=int, default=0)
    ap.add_argument("--max-audio-sec", type=float, default=0.0, help="Optional truncation for memory safety")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-dataparallel", action="store_true")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--skip-oom-batches", action="store_true")
    ap.add_argument("--disable-amp", action="store_true")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--output-dir", default=path_str(MODELS_DIR / "mms1b_aldi_sada"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
