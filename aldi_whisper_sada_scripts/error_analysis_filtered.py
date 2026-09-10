"""
Error analysis for Whisper-ALDi regression checkpoints.

Outputs:
  - by_dialect.csv / by_dialect.png
  - by_duration.csv / by_duration.png
  - by_aldi_range.csv / by_aldi_range.png
  - error_distribution.png
  - scatter_pred_vs_actual.png
  - worst_predictions.csv
  - cross_dataset_comparison.png
  - summary_report.md

Example:
  python scripts/error_analysis.py \
    --checkpoint checkpoints/whisper_aldi_sada_full_medium/checkpoint_epoch5.pt \
    --output-dir analysis/error_analysis_full_sada
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import load_from_disk
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel

from submission_paths import (
    ANALYSIS_DIR,
    CASABLANCA_DIR,
    MEDIASPEECH_DIR,
    MODELS_DIR,
    SADA_DIR,
    path_str,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # aldi_whisper_sada
sys.path.append(SCRIPT_DIR)


class WhisperALDiRegressor(nn.Module):
    def __init__(self, model_id: str, freeze_encoder: bool = False, local_files_only: bool = False):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        ).encoder
        hidden = self.encoder.config.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        enc_outputs = self.encoder(input_features=input_features, attention_mask=attention_mask)
        enc_out = enc_outputs.last_hidden_state
        reduced_mask = torch.nn.functional.interpolate(
            attention_mask.unsqueeze(1).float(),
            size=enc_out.size(1),
            mode="nearest",
        ).squeeze(1)
        mask = reduced_mask.unsqueeze(-1)
        enc_out = enc_out * mask
        pooled = enc_out.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def resolve_path(path: str, manifest_dir: Optional[str] = None) -> str:
    if not isinstance(path, str) or not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = []
    candidates.append(path)
    candidates.append(os.path.join(os.getcwd(), path))
    if manifest_dir:
        candidates.append(os.path.join(manifest_dir, path))
    candidates.append(os.path.join(PROJECT_DIR, path))
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return path


def normalize_score_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ALDi" not in out.columns:
        for col in ["aldi_score", "label", "true_aldi", "aldi"]:
            if col in out.columns:
                out = out.rename(columns={col: "ALDi"})
                break
    return out


def coalesce_columns(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    out = None
    for col in cols:
        if col in df.columns:
            if out is None:
                out = df[col].copy()
            else:
                out = out.where(out.notna(), df[col])
    if out is None:
        out = pd.Series(np.nan, index=df.index)
    return out


def filter_nonspeech_rows(df: pd.DataFrame, dataset_name: str, pattern: str) -> pd.DataFrame:
    if "text" not in df.columns:
        return df
    mask = df["text"].astype(str).str.contains(pattern, regex=True, case=False, na=False)
    removed = int(mask.sum())
    if removed == 0:
        return df
    kept = df.loc[~mask].reset_index(drop=True)
    print(f"[filter] {dataset_name}: removed {removed} non-speech rows | kept {len(kept)}")
    return kept


def infer_best_checkpoint_from_history(checkpoint_dir: str) -> str:
    history_path = os.path.join(checkpoint_dir, "train_history.json")
    if not os.path.isfile(history_path):
        raise FileNotFoundError(f"Missing train history: {history_path}")
    with open(history_path, "r") as f:
        hist = json.load(f)
    if not hist:
        raise RuntimeError(f"Empty train history: {history_path}")
    # Prefer max val_pearson, fallback to min val_loss.
    if "val_pearson" in hist[0]:
        best = max(hist, key=lambda x: x.get("val_pearson", float("-inf")))
    else:
        best = min(hist, key=lambda x: x.get("val_loss", float("inf")))
    epoch = int(best["epoch"])
    ckpt = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Best checkpoint file not found: {ckpt}")
    return ckpt


def resolve_checkpoint(checkpoint_arg: str) -> str:
    # 1) Direct existing file
    direct = resolve_path(checkpoint_arg)
    if os.path.isfile(direct):
        return direct

    # 2) Directory -> infer best
    if os.path.isdir(direct):
        return infer_best_checkpoint_from_history(direct)

    # 3) best.pt pattern in an existing directory
    parent = resolve_path(os.path.dirname(checkpoint_arg) or ".")
    base = os.path.basename(checkpoint_arg)
    if base == "best.pt" and os.path.isdir(parent):
        return infer_best_checkpoint_from_history(parent)

    # 4) Common alias used in notes/prompts.
    if "whisper_medium_full_sada" in checkpoint_arg:
        alias = checkpoint_arg.replace("whisper_medium_full_sada", "whisper_aldi_sada_full_medium")
        alias_path = resolve_path(alias)
        if os.path.isfile(alias_path):
            return alias_path
        alias_parent = resolve_path(os.path.dirname(alias) or ".")
        if os.path.isdir(alias_parent):
            if os.path.basename(alias) == "best.pt":
                return infer_best_checkpoint_from_history(alias_parent)
            if os.path.isfile(alias):
                return alias

    raise FileNotFoundError(
        f"Checkpoint path not found: {checkpoint_arg}. "
        "Pass a valid .pt file, checkpoint directory, or .../best.pt."
    )


@dataclass
class Item:
    input_features: torch.Tensor
    attention_mask: torch.Tensor
    label: torch.Tensor
    meta: Dict


class ManifestAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, fe: WhisperFeatureExtractor, manifest_dir: Optional[str] = None):
        self.df = df.reset_index(drop=True)
        self.fe = fe
        self.target_sr = fe.sampling_rate
        self.manifest_dir = manifest_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Item:
        row = self.df.iloc[idx]
        path = resolve_path(str(row["path"]), self.manifest_dir)
        wav, sr = torchaudio.load(path)
        if wav.dim() > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        wav_np = wav.squeeze(0).numpy()
        feats = self.fe(wav_np, sampling_rate=self.target_sr, return_tensors="pt")
        input_feats = feats.input_features[0]
        attn = torch.ones(input_feats.shape[-1], dtype=torch.long)
        label = torch.tensor(float(row["ALDi"]), dtype=torch.float32)
        meta = row.to_dict()
        meta["resolved_path"] = path
        return Item(input_feats, attn, label, meta)


class CasablancaHFDataset(Dataset):
    def __init__(self, hf_dir: str, split: str, fe: WhisperFeatureExtractor):
        ds_any = load_from_disk(hf_dir)
        self.ds = ds_any[split] if isinstance(ds_any, dict) else ds_any
        self.fe = fe
        self.target_sr = fe.sampling_rate

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Item:
        row = self.ds[idx]
        audio = row["audio"]
        wav = torch.tensor(audio["array"], dtype=torch.float32)
        sr = int(audio["sampling_rate"])
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.target_sr).squeeze(0)
        feats = self.fe(wav.numpy(), sampling_rate=self.target_sr, return_tensors="pt")
        input_feats = feats.input_features[0]
        attn = torch.ones(input_feats.shape[-1], dtype=torch.long)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        meta = {
            "id": row.get("id", row.get("seg_id", idx)),
            "duration_sec": row.get("duration", np.nan),
            "text": row.get("transcription", ""),
            "path": audio.get("path", ""),
        }
        return Item(input_feats, attn, label, meta)


def collate_items(items: List[Item]) -> Dict:
    max_frames = max(it.input_features.shape[-1] for it in items)
    feats, masks, labels, metas = [], [], [], []
    for it in items:
        pad = max_frames - it.input_features.shape[-1]
        if pad > 0:
            feat = torch.nn.functional.pad(it.input_features, (0, pad))
            mask = torch.nn.functional.pad(it.attention_mask, (0, pad))
        else:
            feat = it.input_features
            mask = it.attention_mask
        feats.append(feat)
        masks.append(mask)
        labels.append(it.label)
        metas.append(it.meta)
    return {
        "input_features": torch.stack(feats),
        "attention_mask": torch.stack(masks),
        "labels": torch.stack(labels),
        "metas": metas,
    }


def predict_dataset(
    model: WhisperALDiRegressor,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    dataset_name: str,
) -> pd.DataFrame:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_items,
    )
    rows = []
    model.eval()
    total_batches = len(loader)
    pbar = tqdm(total=total_batches, desc=f"Inference [{dataset_name}]", unit="batch")
    with torch.no_grad():
        for batch in loader:
            feats = batch["input_features"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            preds = model(feats, attention_mask=mask).detach().cpu().numpy()
            for i, meta in enumerate(batch["metas"]):
                row = dict(meta)
                row["dataset"] = dataset_name
                row["true_aldi"] = float(labels[i])
                row["predicted_aldi"] = float(preds[i])
                rows.append(row)
            pbar.update(1)
            pbar.set_postfix({"rows": len(rows)})
    pbar.close()
    df = pd.DataFrame(rows)
    if "dialect" not in df.columns:
        dataset_key = dataset_name.lower()
        if dataset_key == "mediaspeech":
            df["dialect"] = "MSA"
        elif "sada" in dataset_key:
            df["dialect"] = "SADA"
        else:
            df["dialect"] = "unknown"
    return df


def compute_metrics_frame(df: pd.DataFrame) -> Dict[str, float]:
    pred = df["predicted_aldi"].to_numpy()
    true = df["true_aldi"].to_numpy()
    err = pred - true
    out = {
        "n": int(len(df)),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "pearson": safe_pearson(pred, true),
        "spearman": safe_spearman(pred, true),
        "mean_prediction": float(np.mean(pred)),
        "mean_ground_truth": float(np.mean(true)),
        "prediction_std": float(np.std(pred)),
        "error_mean": float(np.mean(err)),
        "error_std": float(np.std(err)),
    }
    return out


def grouped_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {col: keys[i] for i, col in enumerate(group_cols)}
        base.update(compute_metrics_frame(grp))
        rows.append(base)
    out = pd.DataFrame(rows)
    sort_cols = [c for c in group_cols if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def add_duration_bin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "duration_sec" not in out.columns:
        out["duration_sec"] = np.nan
    # Fill from alternative duration col if available.
    if out["duration_sec"].isna().all() and "duration" in out.columns:
        out["duration_sec"] = out["duration"]
    bins = [-np.inf, 5, 10, 15, 20, 30, np.inf]
    labels = ["0-5s", "5-10s", "10-15s", "15-20s", "20-30s", "30s+"]
    out["duration_bin"] = pd.cut(out["duration_sec"], bins=bins, labels=labels, right=False)
    return out


def add_aldi_bin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = [-np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]
    labels = ["<0.0", "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", ">1.0"]
    out["aldi_bin"] = pd.cut(out["true_aldi"], bins=bins, labels=labels, right=False)
    return out


def plot_by_dialect(df: pd.DataFrame, out_png: str) -> None:
    chart = df.sort_values("mae")
    labels = [f"{r.dataset}:{r.dialect}" for _, r in chart.iterrows()]
    x = np.arange(len(chart))
    fig, ax = plt.subplots(figsize=(max(10, len(chart) * 0.5), 5))
    ax.bar(x, chart["mae"].values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("By-Dialect MAE")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_by_duration(df: pd.DataFrame, out_png: str) -> None:
    order = ["0-5s", "5-10s", "10-15s", "15-20s", "20-30s", "30s+"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for dataset, grp in df.groupby("dataset"):
        g = grp.set_index("duration_bin").reindex(order).reset_index()
        axes[0].plot(g["duration_bin"], g["mae"], marker="o", label=dataset)
        axes[1].plot(g["duration_bin"], g["pearson"], marker="o", label=dataset)
    axes[0].set_title("Duration vs MAE")
    axes[1].set_title("Duration vs Pearson")
    axes[0].set_ylabel("MAE")
    axes[1].set_ylabel("Pearson")
    for ax in axes:
        ax.set_xlabel("Duration bin")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_by_aldi_range(df: pd.DataFrame, out_png: str) -> None:
    order = ["<0.0", "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", ">1.0"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for dataset, grp in df.groupby("dataset"):
        g = grp.set_index("aldi_bin").reindex(order).reset_index()
        axes[0].plot(g["aldi_bin"], g["mae"], marker="o", label=dataset)
        axes[1].plot(g["aldi_bin"], g["pearson"], marker="o", label=dataset)
    axes[0].set_title("ALDi Range vs MAE")
    axes[1].set_title("ALDi Range vs Pearson")
    axes[0].set_ylabel("MAE")
    axes[1].set_ylabel("Pearson")
    for ax in axes:
        ax.set_xlabel("Ground-truth ALDi bin")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_error_distribution(df: pd.DataFrame, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, grp in df.groupby("dataset"):
        errs = grp["predicted_aldi"] - grp["true_aldi"]
        ax.hist(errs, bins=50, alpha=0.4, label=dataset, density=True)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_title("Error Distribution (predicted - actual)")
    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"sada_test": "tab:blue", "casablanca": "tab:green", "mediaspeech": "tab:orange"}
    for dataset, grp in df.groupby("dataset"):
        ax.scatter(
            grp["true_aldi"],
            grp["predicted_aldi"],
            s=10,
            alpha=0.35,
            label=dataset,
            c=colors.get(dataset, None),
        )
    min_v = float(np.nanmin([df["true_aldi"].min(), df["predicted_aldi"].min()]))
    max_v = float(np.nanmax([df["true_aldi"].max(), df["predicted_aldi"].max()]))
    ax.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
    ax.set_xlabel("Actual ALDi")
    ax.set_ylabel("Predicted ALDi")
    ax.set_title("Predicted vs Actual ALDi")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_cross_dataset(df_metrics: pd.DataFrame, df_preds: pd.DataFrame, out_png: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df_metrics))
    w = 0.25
    axes[0].bar(x - w, df_metrics["mae"], width=w, label="MAE")
    axes[0].bar(x, df_metrics["pearson"], width=w, label="Pearson")
    axes[0].bar(x + w, df_metrics["spearman"], width=w, label="Spearman")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_metrics["dataset"], rotation=20, ha="right")
    axes[0].set_title("Cross-Dataset Metrics")
    axes[0].legend()

    data = [df_preds[df_preds["dataset"] == ds]["predicted_aldi"].values for ds in df_metrics["dataset"]]
    axes[1].boxplot(data, labels=df_metrics["dataset"], showfliers=False)
    axes[1].set_title("Prediction Distribution by Dataset")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylabel("Predicted ALDi")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def build_summary_report(
    out_path: str,
    ckpt_path: str,
    dataset_metrics: pd.DataFrame,
    by_dialect: pd.DataFrame,
    by_duration: pd.DataFrame,
    by_aldi_range: pd.DataFrame,
    worst_df: pd.DataFrame,
) -> None:
    def md_table(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, r in frame.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join([header, sep] + rows)

    # key summary picks
    best_ds = dataset_metrics.sort_values("pearson", ascending=False).iloc[0]
    worst_ds = dataset_metrics.sort_values("mae", ascending=False).iloc[0]

    # hardest dialect overall (highest MAE with enough samples)
    bd = by_dialect[by_dialect["n"] >= 20] if "n" in by_dialect.columns else by_dialect
    hardest = bd.sort_values("mae", ascending=False).head(5)

    # duration trend clue
    dur_lines = []
    for dataset, grp in by_duration.groupby("dataset"):
        g = grp.dropna(subset=["duration_bin"]).copy()
        if len(g) >= 2:
            short = g[g["duration_bin"] == "0-5s"]["mae"]
            long_ = g[g["duration_bin"] == "20-30s"]["mae"]
            if len(short) and len(long_):
                dur_lines.append(
                    f"- `{dataset}`: MAE 0-5s={float(short.iloc[0]):.4f}, 20-30s={float(long_.iloc[0]):.4f}"
                )

    # ALDi range clue
    range_lines = []
    for dataset, grp in by_aldi_range.groupby("dataset"):
        msa = grp[grp["aldi_bin"] == "0-0.2"]["mae"]
        heavy = grp[grp["aldi_bin"] == "0.8-1.0"]["mae"]
        if len(msa) and len(heavy):
            range_lines.append(
                f"- `{dataset}`: MAE MSA-ish(0-0.2)={float(msa.iloc[0]):.4f}, heavy dialect(0.8-1.0)={float(heavy.iloc[0]):.4f}"
            )

    lines = []
    lines.append("# Error Analysis Summary (Whisper-medium ALDi)")
    lines.append("")
    lines.append(f"- Checkpoint used: `{ckpt_path}`")
    lines.append("")
    lines.append("## Cross-dataset summary")
    lines.append("")
    lines.append(md_table(dataset_metrics))
    lines.append("")
    lines.append(
        f"- Best correlation dataset: `{best_ds['dataset']}` "
        f"(Pearson={best_ds['pearson']:.4f}, Spearman={best_ds['spearman']:.4f})"
    )
    lines.append(
        f"- Highest-error dataset: `{worst_ds['dataset']}` "
        f"(MAE={worst_ds['mae']:.4f})"
    )
    lines.append("")
    lines.append("## Hardest dialects (by MAE)")
    lines.append("")
    if len(hardest):
        lines.append(md_table(hardest[["dataset", "dialect", "n", "mae", "pearson", "spearman"]]))
    else:
        lines.append("- No dialect groups with enough samples.")
    lines.append("")
    lines.append("## Duration effects")
    lines.append("")
    if dur_lines:
        lines.extend(dur_lines)
    else:
        lines.append("- Duration comparison inconclusive from current bins.")
    lines.append("")
    lines.append("## ALDi-range effects")
    lines.append("")
    if range_lines:
        lines.extend(range_lines)
    else:
        lines.append("- ALDi-range comparison inconclusive from current bins.")
    lines.append("")
    lines.append("## Worst-case examples")
    lines.append("")
    lines.append(f"- Top worst rows saved to `worst_predictions.csv` ({len(worst_df)} rows).")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- SADA manifest does not include explicit dialect labels; dialect is set to `SADA`.")
    lines.append("- MediaSpeech is treated as `MSA` dialect in by-dialect grouping.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=path_str(MODELS_DIR / "whisper_aldi_sada_full_medium" / "best.pt"),
        help="Checkpoint .pt, checkpoint directory, or .../best.pt (auto-resolved from train_history).",
    )
    ap.add_argument(
        "--sada-manifest",
        default=path_str(SADA_DIR / "manifest_test_aldi.csv"),
    )
    ap.add_argument(
        "--casablanca-scored-csv",
        default=path_str(CASABLANCA_DIR / "casablanca_aldi_scored.csv"),
    )
    ap.add_argument(
        "--casablanca-hf-dataset",
        default=path_str(CASABLANCA_DIR / "casablanca_audio_aldi"),
        help="load_from_disk() dataset with audio/label/id columns.",
    )
    ap.add_argument(
        "--casablanca-split",
        default="train",
    )
    ap.add_argument(
        "--casablanca-precomputed-preds",
        default=None,
        help="Optional CSV with id,predicted_aldi,true_aldi to skip Casablanca inference.",
    )
    ap.add_argument(
        "--mediaspeech-manifest",
        default=path_str(MEDIASPEECH_DIR / "manifest_mediaspeech_ar_aldi.csv"),
    )
    ap.add_argument("--output-dir", default=path_str(ANALYSIS_DIR / "error_analysis_full_sada_filtered"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument(
        "--filter-nonspeech-tags",
        action="store_true",
        help="Drop rows whose text contains non-speech tags (e.g., [noise], [music], [laughs]).",
    )
    ap.add_argument(
        "--nonspeech-pattern",
        default=r"\[noise\]|\[music\]|\[laughs\]",
        help="Regex used when --filter-nonspeech-tags is enabled.",
    )
    ap.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=0,
        help="For quick smoke runs only. 0 means full datasets.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    ckpt_path = resolve_checkpoint(args.checkpoint)
    print(f"[info] checkpoint: {ckpt_path}")

    if args.local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model_size = ckpt_args.get("model_size", "medium")
    freeze_encoder = ckpt_args.get("freeze_encoder", False)

    hf_kwargs = {"local_files_only": True} if args.local_only or os.environ.get("HF_HUB_OFFLINE") else {}
    fe = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{model_size}", **hf_kwargs)
    model = WhisperALDiRegressor(
        model_id=f"openai/whisper-{model_size}",
        freeze_encoder=freeze_encoder,
        local_files_only=bool(args.local_only or os.environ.get("HF_HUB_OFFLINE")),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pipeline_pbar = tqdm(total=8, desc="Pipeline", unit="step")
    all_frames = []

    # 1) SADA test
    print("[stage] Running SADA inference...")
    sada_manifest_path = resolve_path(args.sada_manifest)
    sada_df = pd.read_csv(sada_manifest_path)
    sada_df = normalize_score_col(sada_df)
    if args.max_samples_per_dataset > 0:
        sada_df = sada_df.head(args.max_samples_per_dataset)
    sada_ds = ManifestAudioDataset(sada_df, fe, manifest_dir=os.path.dirname(sada_manifest_path))
    sada_preds = predict_dataset(
        model,
        sada_ds,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_name="sada_test",
    )
    if args.filter_nonspeech_tags:
        sada_preds = filter_nonspeech_rows(sada_preds, "sada_test", args.nonspeech_pattern)
    all_frames.append(sada_preds)
    print(f"[ok] SADA rows: {len(sada_preds)}")
    pipeline_pbar.update(1)

    # 2) Casablanca
    print("[stage] Running Casablanca inference...")
    casa_meta = pd.read_csv(resolve_path(args.casablanca_scored_csv))
    casa_meta = normalize_score_col(casa_meta)
    if args.casablanca_precomputed_preds:
        casa_pred = pd.read_csv(resolve_path(args.casablanca_precomputed_preds))
        casa_pred = normalize_score_col(casa_pred)
        if "true_aldi" not in casa_pred.columns and "ALDi" in casa_pred.columns:
            casa_pred = casa_pred.rename(columns={"ALDi": "true_aldi"})
        casa = casa_pred.merge(
            casa_meta[["id", "dialect", "audio_path", "duration", "text", "ALDi"]],
            on="id",
            how="left",
        )
        if "true_aldi" not in casa.columns:
            casa["true_aldi"] = coalesce_columns(casa, ["ALDi", "ALDi_y", "ALDi_x"])
        casa["duration_sec"] = coalesce_columns(casa, ["duration_sec", "duration", "duration_y", "duration_x"])
        casa["audio_path"] = coalesce_columns(casa, ["audio_path", "audio_path_y", "audio_path_x"])
        casa["text"] = coalesce_columns(casa, ["text", "text_y", "text_x"]).fillna("")
        casa["dataset"] = "casablanca"
        casa["dialect"] = coalesce_columns(casa, ["dialect", "dialect_y", "dialect_x"]).fillna("unknown")
        if args.max_samples_per_dataset > 0:
            casa = casa.head(args.max_samples_per_dataset)
    else:
        casa_hf_path = resolve_path(args.casablanca_hf_dataset)
        casa_ds = CasablancaHFDataset(casa_hf_path, split=args.casablanca_split, fe=fe)
        if args.max_samples_per_dataset > 0:
            class _SliceDataset(Dataset):
                def __init__(self, base, n):
                    self.base = base
                    self.n = min(len(base), n)
                def __len__(self):
                    return self.n
                def __getitem__(self, idx):
                    return self.base[idx]
            casa_ds_use = _SliceDataset(casa_ds, args.max_samples_per_dataset)
        else:
            casa_ds_use = casa_ds
        casa_pred = predict_dataset(
            model,
            casa_ds_use,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dataset_name="casablanca",
        )
        casa = casa_pred.merge(
            casa_meta[["id", "dialect", "audio_path", "duration", "text", "ALDi"]],
            on="id",
            how="left",
        )
        casa["duration_sec"] = coalesce_columns(casa, ["duration_sec", "duration", "duration_y", "duration_x"])
        casa["audio_path"] = coalesce_columns(casa, ["audio_path", "audio_path_y", "audio_path_x"])
        casa["text"] = coalesce_columns(casa, ["text", "text_y", "text_x"]).fillna("")
        missing_dur = casa["duration_sec"].isna() if "duration_sec" in casa.columns else None
        if missing_dur is not None and missing_dur.any() and "duration" in casa.columns:
            casa.loc[missing_dur, "duration_sec"] = casa.loc[missing_dur, "duration"]
        casa["dialect"] = coalesce_columns(casa, ["dialect", "dialect_y", "dialect_x"]).fillna("unknown")
    if args.filter_nonspeech_tags:
        casa = filter_nonspeech_rows(casa, "casablanca", args.nonspeech_pattern)
    all_frames.append(casa)
    print(f"[ok] Casablanca rows: {len(casa)}")
    pipeline_pbar.update(1)

    # 3) MediaSpeech
    print("[stage] Running MediaSpeech inference...")
    med_manifest_path = resolve_path(args.mediaspeech_manifest)
    med_df = pd.read_csv(med_manifest_path)
    med_df = normalize_score_col(med_df)
    if args.max_samples_per_dataset > 0:
        med_df = med_df.head(args.max_samples_per_dataset)
    med_ds = ManifestAudioDataset(med_df, fe, manifest_dir=os.path.dirname(med_manifest_path))
    med_preds = predict_dataset(
        model,
        med_ds,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_name="mediaspeech",
    )
    med_preds["dialect"] = "MSA"
    if args.filter_nonspeech_tags:
        med_preds = filter_nonspeech_rows(med_preds, "mediaspeech", args.nonspeech_pattern)
    all_frames.append(med_preds)
    print(f"[ok] MediaSpeech rows: {len(med_preds)}")
    pipeline_pbar.update(1)

    # Combined predictions
    print("[stage] Building combined predictions table...")
    df = pd.concat(all_frames, ignore_index=True)
    if "duration_sec" not in df.columns:
        df["duration_sec"] = np.nan
    if "duration" in df.columns:
        mask = df["duration_sec"].isna()
        df.loc[mask, "duration_sec"] = df.loc[mask, "duration"]
    if "id" not in df.columns:
        df["id"] = np.arange(len(df)).astype(str)
    if "resolved_path" not in df.columns and "path" in df.columns:
        df["resolved_path"] = df["path"]

    # Standardized error columns
    df["error"] = df["predicted_aldi"] - df["true_aldi"]
    df["abs_error"] = np.abs(df["error"])

    all_preds_csv = os.path.join(args.output_dir, "all_predictions.csv")
    df.to_csv(all_preds_csv, index=False)
    pipeline_pbar.update(1)

    # 1) by dialect
    print("[stage] Computing by-dialect analysis...")
    by_dialect = grouped_metrics(df, ["dataset", "dialect"])
    by_dialect_csv = os.path.join(args.output_dir, "by_dialect.csv")
    by_dialect.to_csv(by_dialect_csv, index=False)
    plot_by_dialect(by_dialect, os.path.join(args.output_dir, "by_dialect.png"))
    pipeline_pbar.update(1)

    # 2) by duration
    print("[stage] Computing by-duration analysis...")
    df_dur = add_duration_bin(df)
    by_duration = grouped_metrics(df_dur.dropna(subset=["duration_bin"]), ["dataset", "duration_bin"])
    by_duration_csv = os.path.join(args.output_dir, "by_duration.csv")
    by_duration.to_csv(by_duration_csv, index=False)
    plot_by_duration(by_duration, os.path.join(args.output_dir, "by_duration.png"))
    pipeline_pbar.update(1)

    # 3) by ALDi range
    print("[stage] Computing ALDi-range analysis and error plots...")
    df_rng = add_aldi_bin(df)
    by_aldi_range = grouped_metrics(df_rng.dropna(subset=["aldi_bin"]), ["dataset", "aldi_bin"])
    by_aldi_csv = os.path.join(args.output_dir, "by_aldi_range.csv")
    by_aldi_range.to_csv(by_aldi_csv, index=False)
    plot_by_aldi_range(by_aldi_range, os.path.join(args.output_dir, "by_aldi_range.png"))

    # 4) distribution + scatter
    plot_error_distribution(df, os.path.join(args.output_dir, "error_distribution.png"))
    plot_scatter(df, os.path.join(args.output_dir, "scatter_pred_vs_actual.png"))
    pipeline_pbar.update(1)

    # 5) worst predictions
    print("[stage] Writing worst predictions and summary artifacts...")
    worst_cols = [
        "dataset",
        "id",
        "dialect",
        "duration_sec",
        "true_aldi",
        "predicted_aldi",
        "error",
        "abs_error",
        "resolved_path",
        "audio_path",
        "text",
    ]
    keep_cols = [c for c in worst_cols if c in df.columns]
    worst = df.sort_values("abs_error", ascending=False).head(50)[keep_cols].reset_index(drop=True)
    worst.to_csv(os.path.join(args.output_dir, "worst_predictions.csv"), index=False)

    # 6) cross-dataset comparison
    ds_metrics_rows = []
    for dataset, grp in df.groupby("dataset"):
        row = {"dataset": dataset}
        row.update(compute_metrics_frame(grp))
        ds_metrics_rows.append(row)
    ds_metrics = pd.DataFrame(ds_metrics_rows).sort_values("dataset").reset_index(drop=True)
    ds_metrics.to_csv(os.path.join(args.output_dir, "cross_dataset_metrics.csv"), index=False)
    plot_cross_dataset(ds_metrics, df, os.path.join(args.output_dir, "cross_dataset_comparison.png"))

    # summary report
    build_summary_report(
        os.path.join(args.output_dir, "summary_report.md"),
        ckpt_path=ckpt_path,
        dataset_metrics=ds_metrics,
        by_dialect=by_dialect,
        by_duration=by_duration,
        by_aldi_range=by_aldi_range,
        worst_df=worst,
    )
    pipeline_pbar.update(1)
    pipeline_pbar.close()

    print(f"[done] analysis artifacts written to: {args.output_dir}")
    print(f"[saved] {all_preds_csv}")


if __name__ == "__main__":
    main()
