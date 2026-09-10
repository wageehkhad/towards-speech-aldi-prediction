"""
Compare text-based Sentence-ALDi vs speech-model ALDi predictions.

Default workflow:
1) Load predictions from analysis/error_analysis_full_sada/all_predictions.csv
2) Score transcript text with Sentence-ALDi
3) Correlate text_aldi with predicted_aldi
4) Save divergence slices and a scatter plot

Outputs (in --output-dir):
  - merged_with_text_aldi.csv
  - correlation_summary.json
  - scatter_text_vs_speech_aldi.png
  - divergence_text_low_speech_high.csv
  - divergence_text_high_speech_low.csv
  - by_dialect_text_vs_speech.csv (when dialect exists)
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from submission_paths import ANALYSIS_DIR, path_str


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


class ALDiScorer:
    def __init__(self, model_id: str, device: str = "cuda", local_only: bool = False):
        kwargs = {"local_files_only": True} if local_only else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
        resolved_device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.model = self.model.to(resolved_device).eval()
        self.device = resolved_device

    def score(self, texts: List[str], batch_size: int = 32, max_length: int = 256) -> List[float]:
        if not texts:
            return []
        unique_texts = sorted(set(texts))
        text_to_score: Dict[str, float] = {}
        with torch.no_grad():
            for i in tqdm(range(0, len(unique_texts), batch_size), desc="Scoring text ALDi"):
                batch = unique_texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.squeeze(-1).detach().float().cpu().tolist()
                if isinstance(logits, float):
                    logits = [logits]
                for text, score in zip(batch, logits):
                    text_to_score[text] = float(score)
        return [text_to_score[t] for t in texts]


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    pred = df["predicted_aldi"].to_numpy()
    text = df["text_aldi"].to_numpy()
    return {
        "n": int(len(df)),
        "pearson": safe_pearson(text, pred),
        "spearman": safe_spearman(text, pred),
        "mae_text_vs_speech": float(np.mean(np.abs(pred - text))),
        "mean_text_aldi": float(np.mean(text)),
        "mean_predicted_aldi": float(np.mean(pred)),
    }


def plot_scatter(df: pd.DataFrame, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"casablanca": "tab:green", "sada_test": "tab:blue", "mediaspeech": "tab:orange"}
    for ds, grp in df.groupby("dataset"):
        ax.scatter(
            grp["text_aldi"],
            grp["predicted_aldi"],
            s=10,
            alpha=0.35,
            label=ds,
            c=colors.get(ds, None),
        )
    min_v = float(np.nanmin([df["text_aldi"].min(), df["predicted_aldi"].min()]))
    max_v = float(np.nanmax([df["text_aldi"].max(), df["predicted_aldi"].max()]))
    ax.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
    ax.set_xlabel("Text ALDi")
    ax.set_ylabel("Speech predicted ALDi")
    ax.set_title("Text-ALDi vs Speech-ALDi")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions-csv",
        default=path_str(ANALYSIS_DIR / "error_analysis_full_sada" / "all_predictions.csv"),
    )
    ap.add_argument(
        "--dataset",
        default="casablanca",
        help="Dataset subset to analyze: casablanca|sada_test|mediaspeech|all",
    )
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--model-id", default="AMR-KELEG/Sentence-ALDi")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument(
        "--output-dir",
        default=path_str(ANALYSIS_DIR / "text_vs_speech_aldi"),
    )
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--text-low-threshold", type=float, default=0.2)
    ap.add_argument("--speech-high-threshold", type=float, default=0.8)
    ap.add_argument("--text-high-threshold", type=float, default=0.8)
    ap.add_argument("--speech-low-threshold", type=float, default=0.2)
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    df = pd.read_csv(args.predictions_csv)
    if args.dataset.lower() != "all":
        df = df[df["dataset"] == args.dataset].copy()
    if df.empty:
        raise RuntimeError(f"No rows found for dataset='{args.dataset}' in {args.predictions_csv}")

    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column '{args.text_col}' in predictions CSV.")

    # Keep only rows with non-empty text for this experiment.
    text_vals = df[args.text_col].fillna("").astype(str)
    has_text = text_vals.str.strip().ne("")
    dropped = int((~has_text).sum())
    if dropped > 0:
        print(f"[info] dropping {dropped} rows with empty text")
    df = df.loc[has_text].reset_index(drop=True)
    text_vals = df[args.text_col].fillna("").astype(str).tolist()

    if args.local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    scorer = ALDiScorer(
        model_id=args.model_id,
        device=args.device,
        local_only=bool(args.local_only or os.environ.get("HF_HUB_OFFLINE")),
    )
    print(f"[info] scoring text ALDi for {len(df)} rows on {scorer.device}")
    df["text_aldi"] = scorer.score(text_vals, batch_size=args.batch_size, max_length=args.max_length)

    merged_out = os.path.join(args.output_dir, "merged_with_text_aldi.csv")
    df.to_csv(merged_out, index=False)

    summary = {
        "dataset_filter": args.dataset,
        "metrics_overall": compute_metrics(df),
        "text_model": args.model_id,
        "n_rows_scored": int(len(df)),
    }
    by_dataset_rows = []
    for ds, grp in df.groupby("dataset"):
        row = {"dataset": ds}
        row.update(compute_metrics(grp))
        by_dataset_rows.append(row)
    summary["metrics_by_dataset"] = by_dataset_rows

    summary_path = os.path.join(args.output_dir, "correlation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Divergence slices
    low_text_high_speech = df[
        (df["text_aldi"] <= args.text_low_threshold)
        & (df["predicted_aldi"] >= args.speech_high_threshold)
    ].copy()
    low_text_high_speech["gap"] = low_text_high_speech["predicted_aldi"] - low_text_high_speech["text_aldi"]
    low_text_high_speech = low_text_high_speech.sort_values("gap", ascending=False).head(args.top_k)
    low_text_high_speech.to_csv(
        os.path.join(args.output_dir, "divergence_text_low_speech_high.csv"),
        index=False,
    )

    high_text_low_speech = df[
        (df["text_aldi"] >= args.text_high_threshold)
        & (df["predicted_aldi"] <= args.speech_low_threshold)
    ].copy()
    high_text_low_speech["gap"] = high_text_low_speech["text_aldi"] - high_text_low_speech["predicted_aldi"]
    high_text_low_speech = high_text_low_speech.sort_values("gap", ascending=False).head(args.top_k)
    high_text_low_speech.to_csv(
        os.path.join(args.output_dir, "divergence_text_high_speech_low.csv"),
        index=False,
    )

    # Optional by-dialect breakdown.
    if "dialect" in df.columns:
        rows = []
        for key, grp in df.groupby(["dataset", "dialect"]):
            row = {"dataset": key[0], "dialect": key[1]}
            row.update(compute_metrics(grp))
            rows.append(row)
        if rows:
            pd.DataFrame(rows).sort_values(["dataset", "dialect"]).to_csv(
                os.path.join(args.output_dir, "by_dialect_text_vs_speech.csv"),
                index=False,
            )

    plot_scatter(df, os.path.join(args.output_dir, "scatter_text_vs_speech_aldi.png"))

    print("[done] text-vs-speech analysis complete")
    print(f"[saved] {merged_out}")
    print(f"[saved] {summary_path}")
    print(
        f"[saved] {os.path.join(args.output_dir, 'divergence_text_low_speech_high.csv')}"
    )
    print(
        f"[saved] {os.path.join(args.output_dir, 'divergence_text_high_speech_low.csv')}"
    )


if __name__ == "__main__":
    main()
