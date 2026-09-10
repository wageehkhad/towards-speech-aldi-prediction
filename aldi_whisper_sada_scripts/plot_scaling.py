"""
Plot RMSE against training data volume for the direct ALDi models.

Input:  results/experiments.csv, one row per (model, train_hours, test_set) run.
Output: results/scaling_figure_rmse.pdf and .png

Expected columns:
    run_id        free-form identifier for the training run
    model         model label used in the legend, e.g. whisper-medium, mms-1b
    train_hours   hours of SADA training audio used for that run
    test_set      sada_test, casablanca or mediaspeech
    rmse          RMSE of the predicted ALDi scores on that test set
    mae           optional
    pearson       optional
    spearman      optional
    checkpoint    optional path to the checkpoint the row was scored from
    notes         optional

Rows with a blank rmse are skipped. Any test set present in the file is plotted
as its own panel, so a partially filled file still produces a figure.
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from submission_paths import RESULTS_DIR, path_str

REQUIRED = ["model", "train_hours", "test_set", "rmse"]

TEST_SET_LABELS = {
    "sada_test": "SADA (test set)",
    "casablanca": "Casablanca",
    "mediaspeech": "MediaSpeech",
}

MARKERS = ["o", "s", "^", "D", "v", "P"]

# Panel order follows Table 1 rather than whatever order the CSV happens to use.
TEST_SET_ORDER = ["sada_test", "mediaspeech", "casablanca"]


def load_runs(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Experiments file not found: {}".format(csv_path))

    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Experiments file missing columns in {}: {}".format(csv_path, missing))

    df = df.dropna(subset=["rmse", "train_hours", "model", "test_set"])
    df["train_hours"] = pd.to_numeric(df["train_hours"], errors="coerce")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df = df.dropna(subset=["train_hours", "rmse"])

    if df.empty:
        raise RuntimeError(
            "No usable rows in {}. Add one row per training run with "
            "model, train_hours, test_set and rmse.".format(csv_path)
        )

    return df.sort_values(["test_set", "model", "train_hours"])


def plot(df: pd.DataFrame, out_path: Path, annotate: bool) -> None:
    present = list(dict.fromkeys(df["test_set"]))
    test_sets = [t for t in TEST_SET_ORDER if t in present]
    test_sets += [t for t in present if t not in TEST_SET_ORDER]
    fig, axes = plt.subplots(
        1, len(test_sets), figsize=(4.2 * len(test_sets), 3.6), sharey=True, squeeze=False
    )
    axes = axes[0]

    models = list(dict.fromkeys(df["model"]))
    marker_for = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(models)}

    for ax, ts in zip(axes, test_sets):
        sub = df[df["test_set"] == ts]
        for idx, model in enumerate(models):
            rows = sub[sub["model"] == model]
            if rows.empty:
                continue
            ax.plot(
                rows["train_hours"],
                rows["rmse"],
                marker=marker_for[model],
                linewidth=1.6,
                markersize=5,
                label=model,
            )
            if annotate:
                for _, r in rows.iterrows():
                    ax.annotate(
                        "{:.3f}".format(r["rmse"]),
                        (r["train_hours"], r["rmse"]),
                        textcoords="offset points",
                        xytext=(0, 7 if idx % 2 == 0 else -12),
                        fontsize=7,
                        ha="center",
                    )
        ax.set_xscale("log")
        ax.set_xlabel("Training hours")
        ax.set_title(TEST_SET_LABELS.get(ts, ts), fontsize=10)
        ax.grid(True, which="both", alpha=0.3, linewidth=0.5)

    axes[0].set_ylabel("RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=8, frameon=False)

    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        target = out_path.with_suffix(suffix)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        print("[ok] wrote {}".format(target))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default=path_str(RESULTS_DIR / "experiments.csv"))
    ap.add_argument("--output", default=path_str(RESULTS_DIR / "scaling_figure_rmse.pdf"))
    ap.add_argument("--no-annotate", action="store_true", help="Omit RMSE value labels")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    df = load_runs(args.experiments)

    print("[info] plotting {} run(s):".format(len(df)))
    for ts, grp in df.groupby("test_set"):
        for model, rows in grp.groupby("model"):
            hours = ", ".join("{:g}h".format(h) for h in rows["train_hours"])
            print("[info]   {:<12} {:<16} {}".format(ts, model, hours))

    plot(df, Path(args.output), annotate=not args.no_annotate)


if __name__ == "__main__":
    main()
