#!/usr/bin/env python3
"""
Compare:
1) Baseline1 pretrained ASR -> Sentence-ALDi
2) Baseline2 finetuned ASR -> Sentence-ALDi
3) Direct ALDi speech model
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from submission_paths import ANALYSIS_DIR, RESULTS_DIR, path_str


DATASETS = ["sada_test", "casablanca", "mediaspeech"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_baseline_metrics(summary_json: str) -> Dict[str, Dict[str, float]]:
    with open(summary_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "datasets" in payload:
        return payload["datasets"]
    return payload


def load_direct_metrics(path: str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(path)
    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        ds = str(row["dataset"])
        rmse = float(row["rmse"]) if "rmse" in df.columns else float(np.sqrt(float(row["mse"])))
        out[ds] = {
            "pearson": float(row["pearson"]),
            "spearman": float(row["spearman"]),
            "mae": float(row["mae"]),
            "rmse": rmse,
        }
    return out


def build_comparison_table(
    base1: Dict[str, Dict[str, float]],
    base2: Dict[str, Dict[str, float]],
    direct: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    rows = []
    mapping = [
        ("baseline1_pretrained_asr", base1),
        ("baseline2_finetuned_asr", base2),
        ("direct_aldi_model", direct),
    ]
    for approach, m in mapping:
        row: Dict[str, float] = {"approach": approach}
        for ds in DATASETS:
            val = m.get(ds, {})
            row["{}_pearson".format(ds)] = float(val.get("pearson", float("nan")))
            row["{}_spearman".format(ds)] = float(val.get("spearman", float("nan")))
            row["{}_mae".format(ds)] = float(val.get("mae", float("nan")))
            rmse = val.get("rmse", None)
            if rmse is None and "mse" in val:
                rmse = float(np.sqrt(float(val["mse"])))
            row["{}_rmse".format(ds)] = float(rmse) if rmse is not None else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = float(np.mean(sample))
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1 - alpha / 2.0)),
    )


def load_direct_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "id" in df.columns and "sample_id" not in df.columns:
        df = df.rename(columns={"id": "sample_id"})
    needed = ["dataset", "sample_id", "true_aldi", "predicted_aldi"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError("Direct predictions missing columns: {}".format(miss))
    df["sample_id"] = df["sample_id"].astype(str)
    return df[needed].copy()


def load_baseline_predictions(path: str, dataset: str) -> pd.DataFrame:
    file_path = os.path.join(path, "predictions_{}.csv".format(dataset))
    df = pd.read_csv(file_path).copy()
    needed = ["sample_id", "true_aldi", "predicted_aldi"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError("Baseline predictions missing columns in {}: {}".format(file_path, miss))
    df["sample_id"] = df["sample_id"].astype(str)
    return df[needed].copy()


def paired_tests(
    baseline_dir: str,
    baseline_name: str,
    direct_df: pd.DataFrame,
    out_lines: List[str],
    bootstrap_samples: int,
    seed: int,
) -> None:
    out_lines.append("## {}".format(baseline_name))
    for ds in DATASETS:
        base_df = load_baseline_predictions(baseline_dir, ds)
        ref = direct_df[direct_df["dataset"] == ds][["sample_id", "predicted_aldi"]].rename(
            columns={"predicted_aldi": "pred_direct"}
        )
        cur = base_df[["sample_id", "true_aldi", "predicted_aldi"]].rename(
            columns={"predicted_aldi": "pred_baseline"}
        )
        merged = cur.merge(ref, on="sample_id", how="inner")
        if merged.empty:
            out_lines.append("- {}: no overlapping IDs".format(ds))
            continue

        err_baseline = np.abs(merged["pred_baseline"].to_numpy() - merged["true_aldi"].to_numpy())
        err_direct = np.abs(merged["pred_direct"].to_numpy() - merged["true_aldi"].to_numpy())
        diff = err_baseline - err_direct
        # Positive diff = direct model has lower error.
        mean_improve = float(np.mean(diff))

        if len(diff) > 1:
            t_res = ttest_rel(err_baseline, err_direct, nan_policy="omit")
            p_val = float(t_res.pvalue)
        else:
            p_val = float("nan")
        ci_low, ci_high = bootstrap_mean_ci(diff, n_boot=bootstrap_samples, seed=seed)
        out_lines.append(
            "- {}: n={} mean_abs_error_improvement(direct-baseline)={:.6f} "
            "95%CI=[{:.6f},{:.6f}] paired_t_p={:.6g}".format(
                ds, len(diff), mean_improve, ci_low, ci_high, p_val
            )
        )
    out_lines.append("")


def plot_comparison_chart(comparison_df: pd.DataFrame, out_png: str) -> None:
    approaches = comparison_df["approach"].tolist()
    x = np.arange(len(DATASETS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, approach in enumerate(approaches):
        vals = []
        row = comparison_df[comparison_df["approach"] == approach].iloc[0]
        for ds in DATASETS:
            vals.append(float(row["{}_pearson".format(ds)]))
        ax.bar(x + (i - 1) * width, vals, width=width, label=approach)

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("Pearson")
    ax.set_title("Pearson Correlation Comparison by Approach")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline1-dir",
        default=path_str(RESULTS_DIR / "baseline1_pretrained_asr"),
    )
    ap.add_argument(
        "--baseline2-dir",
        default=path_str(RESULTS_DIR / "baseline2_finetuned_asr"),
    )
    ap.add_argument(
        "--direct-metrics-csv",
        default=path_str(ANALYSIS_DIR / "error_analysis_full_sada" / "cross_dataset_metrics.csv"),
    )
    ap.add_argument(
        "--direct-predictions-csv",
        default=path_str(ANALYSIS_DIR / "error_analysis_full_sada" / "all_predictions.csv"),
    )
    ap.add_argument("--bootstrap-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--output-dir",
        default=path_str(RESULTS_DIR / "baseline_comparison"),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    b1 = load_baseline_metrics(os.path.join(args.baseline1_dir, "metrics_summary.json"))
    b2 = load_baseline_metrics(os.path.join(args.baseline2_dir, "metrics_summary.json"))
    direct = load_direct_metrics(args.direct_metrics_csv)

    comparison = build_comparison_table(b1, b2, direct)
    table_path = os.path.join(args.output_dir, "comparison_table.csv")
    comparison.to_csv(table_path, index=False)

    chart_path = os.path.join(args.output_dir, "comparison_chart.png")
    plot_comparison_chart(comparison, chart_path)

    direct_df = load_direct_predictions(args.direct_predictions_csv)
    lines: List[str] = [
        "# Paired Statistical Tests",
        "",
        "Metric: absolute error vs ground-truth ALDi",
        "Positive improvement means direct ALDi model has lower error than baseline.",
        "",
    ]
    paired_tests(
        baseline_dir=args.baseline1_dir,
        baseline_name="baseline1_pretrained_asr",
        direct_df=direct_df,
        out_lines=lines,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    paired_tests(
        baseline_dir=args.baseline2_dir,
        baseline_name="baseline2_finetuned_asr",
        direct_df=direct_df,
        out_lines=lines,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    tests_path = os.path.join(args.output_dir, "statistical_tests.txt")
    with open(tests_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("[saved] {}".format(table_path))
    print("[saved] {}".format(chart_path))
    print("[saved] {}".format(tests_path))


if __name__ == "__main__":
    main()
