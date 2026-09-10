#!/usr/bin/env python3
"""
Analyze how ASR word error rate (WER) relates to ALDi prediction quality.

Inputs:
- aligned per-utterance predictions for baseline2 (B2) and the direct model

Outputs:
- casablanca_dialect_wer_rmse.csv
- wer_bins_rmse.csv
- casablanca_dialect_wer_vs_rmse.png
- wer_bins_rmse.png
- summary.txt
"""

import argparse
import os
import re
import string
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from submission_paths import RESULTS_DIR, path_str


# WER is computed here rather than with jiwer so that reference and hypothesis
# go through the same Arabic-aware normalisation before alignment.
ASCII_PUNCT = string.punctuation
AR_PUNCT = "،؛؟«»ـ…“”‘’"
PUNCT_RE = re.compile("[{}]".format(re.escape(ASCII_PUNCT + AR_PUNCT)))
SPACE_RE = re.compile(r"\s+")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_text(text: object) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).strip()
    # Drop the unclear-speech annotation marker so it is not counted as a
    # reference token the ASR was expected to produce.
    s = s.replace("#غير_واضح", " ")
    s = PUNCT_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)
    for i, tok_a in enumerate(a, start=1):
        cur[0] = i
        for j, tok_b in enumerate(b, start=1):
            cost = 0 if tok_a == tok_b else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, cur = cur, prev
    return prev[-1]


# Returns NaN for an empty reference so those rows drop out of the WER means
# instead of contributing a division by zero.
def word_error_rate(reference: object, hypothesis: object) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return float("nan")
    return float(levenshtein_distance(ref, hyp) / max(1, len(ref)))


def compute_rmse(errors: Iterable[float]) -> float:
    arr = np.asarray(list(errors), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr ** 2)))


def add_wer_and_errors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["wer"] = [
        word_error_rate(ref, hyp)
        for ref, hyp in zip(out["reference_text_b2"], out["asr_transcript_b2"])
    ]
    out["sq_err_b2"] = (out["pred_aldi_b2"] - out["true_aldi"]) ** 2
    out["sq_err_direct"] = (out["pred_aldi_direct"] - out["true_aldi"]) ** 2
    out["abs_err_b2"] = np.abs(out["pred_aldi_b2"] - out["true_aldi"])
    out["abs_err_direct"] = np.abs(out["pred_aldi_direct"] - out["true_aldi"])
    out["b2_better"] = out["abs_err_b2"] < out["abs_err_direct"]
    out["direct_better"] = out["abs_err_direct"] < out["abs_err_b2"]
    return out


def casablanca_by_dialect(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    casa = df[df["dataset"] == "casablanca"].copy()
    for dialect, grp in casa.groupby("dialect_b2"):
        rows.append(
            {
                "dialect": dialect,
                "n": int(len(grp)),
                "mean_wer": float(np.nanmean(grp["wer"])),
                "median_wer": float(np.nanmedian(grp["wer"])),
                "rmse_b2": compute_rmse(grp["pred_aldi_b2"] - grp["true_aldi"]),
                "rmse_direct": compute_rmse(grp["pred_aldi_direct"] - grp["true_aldi"]),
                "mae_b2": float(np.mean(grp["abs_err_b2"])),
                "mae_direct": float(np.mean(grp["abs_err_direct"])),
                "mean_true_aldi": float(np.mean(grp["true_aldi"])),
                "mean_pred_b2": float(np.mean(grp["pred_aldi_b2"])),
                "mean_pred_direct": float(np.mean(grp["pred_aldi_direct"])),
                "b2_win_rate": float(np.mean(grp["b2_better"])),
                "direct_win_rate": float(np.mean(grp["direct_better"])),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_wer").reset_index(drop=True)


def wer_bins_table(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 0.1, 0.25, 0.5, 0.75, 1.0, np.inf]
    labels = ["0.00-0.10", "0.10-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00", "1.00+"]
    out = df.copy()
    out["wer_bin"] = pd.cut(out["wer"], bins=bins, labels=labels, right=False)

    rows = []
    for (dataset, wer_bin), grp in out.groupby(["dataset", "wer_bin"], dropna=True):
        rows.append(
            {
                "dataset": dataset,
                "wer_bin": str(wer_bin),
                "n": int(len(grp)),
                "mean_wer": float(np.nanmean(grp["wer"])),
                "rmse_b2": compute_rmse(grp["pred_aldi_b2"] - grp["true_aldi"]),
                "rmse_direct": compute_rmse(grp["pred_aldi_direct"] - grp["true_aldi"]),
                "mae_b2": float(np.mean(grp["abs_err_b2"])),
                "mae_direct": float(np.mean(grp["abs_err_direct"])),
                "b2_win_rate": float(np.mean(grp["b2_better"])),
                "direct_win_rate": float(np.mean(grp["direct_better"])),
            }
        )

    for wer_bin, grp in out.groupby("wer_bin", dropna=True):
        rows.append(
            {
                "dataset": "ALL",
                "wer_bin": str(wer_bin),
                "n": int(len(grp)),
                "mean_wer": float(np.nanmean(grp["wer"])),
                "rmse_b2": compute_rmse(grp["pred_aldi_b2"] - grp["true_aldi"]),
                "rmse_direct": compute_rmse(grp["pred_aldi_direct"] - grp["true_aldi"]),
                "mae_b2": float(np.mean(grp["abs_err_b2"])),
                "mae_direct": float(np.mean(grp["abs_err_direct"])),
                "b2_win_rate": float(np.mean(grp["b2_better"])),
                "direct_win_rate": float(np.mean(grp["direct_better"])),
            }
        )

    return pd.DataFrame(rows).sort_values(["dataset", "wer_bin"]).reset_index(drop=True)


def plot_casablanca_dialect(dialect_df: pd.DataFrame, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(dialect_df["mean_wer"], dialect_df["rmse_b2"], label="B2", s=80)
    ax.scatter(dialect_df["mean_wer"], dialect_df["rmse_direct"], label="Direct", s=80)
    for _, row in dialect_df.iterrows():
        ax.annotate(str(row["dialect"]), (row["mean_wer"], row["rmse_b2"]), xytext=(4, 4), textcoords="offset points")
        ax.annotate(str(row["dialect"]), (row["mean_wer"], row["rmse_direct"]), xytext=(4, -10), textcoords="offset points")
    ax.set_xlabel("Mean WER")
    ax.set_ylabel("RMSE")
    ax.set_title("Casablanca Dialect Mean WER vs RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_wer_bins(bin_df: pd.DataFrame, out_png: str) -> None:
    all_df = bin_df[bin_df["dataset"] == "ALL"].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(all_df))
    w = 0.35
    ax.bar(x - w / 2, all_df["rmse_b2"], width=w, label="B2")
    ax.bar(x + w / 2, all_df["rmse_direct"], width=w, label="Direct")
    ax.set_xticks(x)
    ax.set_xticklabels(all_df["wer_bin"], rotation=25, ha="right")
    ax.set_xlabel("WER bin")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by WER Bin (All Datasets)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def build_summary(dialect_df: pd.DataFrame, bin_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    lines.append("WER vs ALDi error analysis")
    lines.append("")

    casa = dialect_df.sort_values("mean_wer")
    lines.append("Casablanca dialects ordered by mean WER:")
    for _, row in casa.iterrows():
        lines.append(
            "{}: n={} mean_wer={:.3f} rmse_b2={:.3f} rmse_direct={:.3f}".format(
                row["dialect"], int(row["n"]), row["mean_wer"], row["rmse_b2"], row["rmse_direct"]
            )
        )
    lines.append("")

    all_bins = bin_df[bin_df["dataset"] == "ALL"].copy()
    lines.append("WER bins across all datasets:")
    for _, row in all_bins.iterrows():
        lines.append(
            "{}: n={} mean_wer={:.3f} rmse_b2={:.3f} rmse_direct={:.3f} b2_win_rate={:.3f} direct_win_rate={:.3f}".format(
                row["wer_bin"],
                int(row["n"]),
                row["mean_wer"],
                row["rmse_b2"],
                row["rmse_direct"],
                row["b2_win_rate"],
                row["direct_win_rate"],
            )
        )
    return lines


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--aligned-csv",
        default=path_str(RESULTS_DIR / "baseline_comparison" / "aligned_b2_vs_direct_whisper_medium_full_all3_both_only.csv"),
    )
    ap.add_argument(
        "--output-dir",
        default=path_str(RESULTS_DIR / "baseline_comparison" / "wer_analysis"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    df = pd.read_csv(args.aligned_csv)
    df = add_wer_and_errors(df)

    dialect_df = casablanca_by_dialect(df)
    bins_df = wer_bins_table(df)

    dialect_csv = os.path.join(args.output_dir, "casablanca_dialect_wer_rmse.csv")
    bins_csv = os.path.join(args.output_dir, "wer_bins_rmse.csv")
    dialect_png = os.path.join(args.output_dir, "casablanca_dialect_wer_vs_rmse.png")
    bins_png = os.path.join(args.output_dir, "wer_bins_rmse.png")
    summary_txt = os.path.join(args.output_dir, "summary.txt")

    dialect_df.to_csv(dialect_csv, index=False)
    bins_df.to_csv(bins_csv, index=False)
    plot_casablanca_dialect(dialect_df, dialect_png)
    plot_wer_bins(bins_df, bins_png)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(build_summary(dialect_df, bins_df)) + "\n")

    print("[saved] {}".format(dialect_csv))
    print("[saved] {}".format(bins_csv))
    print("[saved] {}".format(dialect_png))
    print("[saved] {}".format(bins_png))
    print("[saved] {}".format(summary_txt))


if __name__ == "__main__":
    main()
