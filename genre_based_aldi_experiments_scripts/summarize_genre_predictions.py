#!/usr/bin/env python3
"""
Aggregate clip-level ALDi predictions into per-source and per-genre summaries.

Produces two passes over the same predictions: the full pooled set, and a
Saudi/Gulf control set. Outputs are box plots, violin plots, per-genre and
per-source tables, duration breakdowns, and an ordering report.
"""

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_ORDER = [
    "religious",
    "news",
    "drama",
    "talkshow",
    "podcast",
    "street_interview",
]

# sources.csv labels podcasts and TV by dialect region; the reported analysis
# collapses them into the six genres, so remap_genre folds these back together.
PODCAST_GENRES = {"podcast_gulf", "podcast_maghrebi", "podcast_levant"}
# Sources restricted to a single dialect region. Re-running the genre analysis
# over only these holds dialect roughly constant, so a genre separation that
# survives here cannot be explained by the model tracking accent instead.
GULF_CONTROL_SOURCE_IDS = {
    "news_01",
    "news_02",
    "news_03",
    "news_04",
    "religious_01",
    "religious_02",
    "religious_03",
    "drama_02",
    "drama_03",
    "drama_04",
    "talkshow_01",
    "talkshow_08",
    "talkshow_09",
    "podcast_gulf_01",
    "podcast_gulf_02",
    "podcast_gulf_03",
    "street_saudi_01",
    "street_saudi_02",
    "street_saudi_03",
    "street_saudi_short_01",
}

DIALECT_BUCKET_ORDER = [
    "msa",
    "religious_formal",
    "gulf",
    "levant",
    "egyptian",
    "maghrebi",
]


def parse_args():
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results" / "genre_based_aldi"
    ap = argparse.ArgumentParser(description="Aggregate clip-level ALDi predictions by source and genre.")
    ap.add_argument("--input-csv", default=str(results_dir / "clip_predictions.csv"))
    ap.add_argument("--output-dir", default=str(results_dir))
    return ap.parse_args()


def mean(values):
    return statistics.fmean(values) if values else math.nan


def std(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def remap_genre(genre: str) -> str:
    if genre in PODCAST_GENRES:
        return "podcast"
    if genre == "podcast_egypt":
        return "podcast"
    if genre == "tv_show":
        return "drama"
    return genre


# Collapses the free-text dialect_region values in sources.csv into the buckets
# used for grouping. Anything unrecognised is slugified rather than dropped.
def remap_dialect_bucket(dialect_region: str) -> str:
    normalized = dialect_region.strip().lower()
    mapping = {
        "msa": "msa",
        "formal recited register": "religious_formal",
        "gulf": "gulf",
        "saudi": "gulf",
        "kuwaiti": "gulf",
        "uae/gulf": "gulf",
        "saudi/gulf": "gulf",
        "levantine": "levant",
        "lebanese": "levant",
        "jordanian": "levant",
        "egyptian": "egyptian",
        "moroccan": "maghrebi",
    }
    if normalized in mapping:
        return mapping[normalized]
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "unknown"


def transform_rows(rows, allowed_source_ids=None, excluded_source_ids=None):
    excluded_source_ids = excluded_source_ids or set()
    transformed = []
    for row in rows:
        if row["source_id"] in excluded_source_ids:
            continue
        if allowed_source_ids is not None and row["source_id"] not in allowed_source_ids:
            continue
        new_row = dict(row)
        new_row["genre"] = remap_genre(new_row["genre"])
        new_row["dialect_bucket"] = remap_dialect_bucket(new_row["dialect_region"])
        transformed.append(new_row)
    return transformed


def summarize_rows(rows, group_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(float(row["predicted_aldi"]))

    summary = []
    for key, values in grouped.items():
        summary.append(
            {
                group_key: key,
                "n_clips": len(values),
                "mean_predicted_aldi": round(mean(values), 6),
                "median_predicted_aldi": round(statistics.median(values), 6),
                "std_predicted_aldi": round(std(values), 6),
                "min_predicted_aldi": round(min(values), 6),
                "max_predicted_aldi": round(max(values), 6),
            }
        )
    return sorted(summary, key=lambda row: row["mean_predicted_aldi"])


def summarize_durations(rows, group_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        bucket = grouped.setdefault(
            key,
            {
                **{group_key: row[group_key] for group_key in group_keys},
                "n_clips": 0,
                "n_sources": set(),
                "total_duration_sec": 0.0,
            },
        )
        bucket["n_clips"] += 1
        bucket["n_sources"].add(row["source_id"])
        bucket["total_duration_sec"] += float(row["duration_sec"])

    summary = []
    for item in grouped.values():
        total_sec = item["total_duration_sec"]
        summary.append(
            {
                **{group_key: item[group_key] for group_key in group_keys},
                "n_sources": len(item["n_sources"]),
                "n_clips": item["n_clips"],
                "total_duration_sec": round(total_sec, 3),
                "total_duration_min": round(total_sec / 60.0, 3),
            }
        )

    def sort_key(row):
        key = []
        for group_key in group_keys:
            value = row[group_key]
            if group_key == "genre":
                key.append(DISPLAY_ORDER.index(value) if value in DISPLAY_ORDER else len(DISPLAY_ORDER))
            elif group_key == "dialect_bucket":
                key.append(
                    DIALECT_BUCKET_ORDER.index(value)
                    if value in DIALECT_BUCKET_ORDER
                    else len(DIALECT_BUCKET_ORDER)
                )
            else:
                key.append(value)
            key.append(value)
        return tuple(key)

    return sorted(summary, key=sort_key)


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows, output_path: Path):
    by_genre = defaultdict(list)
    for row in rows:
        by_genre[row["genre"]].append(float(row["predicted_aldi"]))

    ordered = [genre for genre in DISPLAY_ORDER if genre in by_genre]
    data = [by_genre[genre] for genre in ordered]
    labels = [genre.replace("_", "\n") for genre in ordered]

    plt.figure(figsize=(11, 6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Predicted ALDi")
    plt.title("Genre-based direct Whisper-ALDi predictions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_mean_plot(rows, output_path: Path):
    by_genre = defaultdict(list)
    for row in rows:
        by_genre[row["genre"]].append(float(row["predicted_aldi"]))

    ordered = [genre for genre in DISPLAY_ORDER if genre in by_genre]
    means = [mean(by_genre[genre]) for genre in ordered]
    labels = [genre.replace("_", "\n") for genre in ordered]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(labels, means)
    plt.ylabel("Mean predicted ALDi")
    plt.title("Mean predicted ALDi by genre")
    plt.ylim(min(0.0, min(means) - 0.05), max(means) + 0.05)
    for bar, value in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_violin_plot(rows, output_path: Path):
    by_genre = defaultdict(list)
    for row in rows:
        by_genre[row["genre"]].append(float(row["predicted_aldi"]))

    ordered = [genre for genre in DISPLAY_ORDER if genre in by_genre]
    data = [by_genre[genre] for genre in ordered]
    labels = [genre.replace("_", "\n") for genre in ordered]

    plt.figure(figsize=(11, 6))
    parts = plt.violinplot(data, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.6)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("Predicted ALDi")
    plt.title("Genre-based direct Whisper-ALDi violin plot")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_duration_heatmap(duration_rows, output_path: Path):
    dialects = [bucket for bucket in DIALECT_BUCKET_ORDER if any(row["dialect_bucket"] == bucket for row in duration_rows)]
    genres = [genre for genre in DISPLAY_ORDER if any(row["genre"] == genre for row in duration_rows)]
    if not dialects or not genres:
        return

    values = np.zeros((len(dialects), len(genres)), dtype=float)
    for row in duration_rows:
        i = dialects.index(row["dialect_bucket"])
        j = genres.index(row["genre"])
        values[i, j] = float(row["total_duration_min"])

    plt.figure(figsize=(10, 5.5))
    im = plt.imshow(values, cmap="YlGnBu", aspect="auto")
    plt.xticks(range(len(genres)), [genre.replace("_", "\n") for genre in genres])
    plt.yticks(range(len(dialects)), [dialect.replace("_", "\n") for dialect in dialects])
    plt.title("Duration heatmap by dialect and genre")
    plt.xlabel("Genre")
    plt.ylabel("Dialect bucket")
    cbar = plt.colorbar(im)
    cbar.set_label("Minutes")

    for i in range(len(dialects)):
        for j in range(len(genres)):
            plt.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_report(path: Path, by_genre, comparable, title: str):
    means = {row["genre"]: row["mean_predicted_aldi"] for row in by_genre}
    order_pairs = list(zip(comparable, comparable[1:]))
    monotonic = all(means[left] < means[right] for left, right in order_pairs) if order_pairs else False
    report_lines = [
        title,
        "",
        "Mean predictions by genre (ascending):",
    ]
    for row in by_genre:
        report_lines.append(f"- {row['genre']}: {row['mean_predicted_aldi']:.6f} (n={row['n_clips']})")
    report_lines.extend(
        [
            "",
            "Main ordering check:",
            f"- comparable sequence: {' < '.join(comparable) if comparable else 'n/a'}",
            f"- monotonic increasing: {monotonic}",
        ]
    )
    path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run_analysis(rows, output_dir: Path, stem_suffix: str, report_title: str):
    by_source = summarize_rows(rows, "source_id")
    by_genre = summarize_rows(rows, "genre")
    duration_by_genre = summarize_durations(rows, ["genre"])
    duration_by_dialect = summarize_durations(rows, ["dialect_bucket"])
    duration_by_dialect_and_genre = summarize_durations(rows, ["dialect_bucket", "genre"])
    summary_by_source_path = output_dir / f"summary_by_source{stem_suffix}.csv"
    summary_by_genre_path = output_dir / f"summary_by_genre{stem_suffix}.csv"
    duration_by_genre_path = output_dir / f"duration_by_genre{stem_suffix}.csv"
    duration_by_dialect_path = output_dir / f"duration_by_dialect{stem_suffix}.csv"
    duration_by_dialect_and_genre_path = output_dir / f"duration_by_dialect_and_genre{stem_suffix}.csv"
    boxplot_path = output_dir / f"genre_boxplot{stem_suffix}.png"
    mean_plot_path = output_dir / f"genre_mean_plot{stem_suffix}.png"
    violin_plot_path = output_dir / f"genre_violin_plot{stem_suffix}.png"
    duration_heatmap_path = output_dir / f"duration_heatmap{stem_suffix}.png"
    report_path = output_dir / f"ordering_report{stem_suffix}.txt"

    write_csv(summary_by_source_path, by_source)
    write_csv(summary_by_genre_path, by_genre)
    write_csv(duration_by_genre_path, duration_by_genre)
    write_csv(duration_by_dialect_path, duration_by_dialect)
    write_csv(duration_by_dialect_and_genre_path, duration_by_dialect_and_genre)
    make_plot(rows, boxplot_path)
    make_mean_plot(rows, mean_plot_path)
    make_violin_plot(rows, violin_plot_path)
    make_duration_heatmap(duration_by_dialect_and_genre, duration_heatmap_path)
    write_report(
        report_path,
        by_genre,
        comparable=["news", "drama", "talkshow", "podcast"],
        title=report_title,
    )
    return {
        "source": summary_by_source_path,
        "genre": summary_by_genre_path,
        "duration_by_genre": duration_by_genre_path,
        "duration_by_dialect": duration_by_dialect_path,
        "duration_by_dialect_and_genre": duration_by_dialect_and_genre_path,
        "boxplot": boxplot_path,
        "mean_plot": mean_plot_path,
        "violin_plot": violin_plot_path,
        "duration_heatmap": duration_heatmap_path,
        "report": report_path,
    }


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No prediction rows found in {input_csv}")

    full_rows = transform_rows(rows)
    gulf_control_rows = transform_rows(
        rows,
        allowed_source_ids=GULF_CONTROL_SOURCE_IDS,
    )

    full_outputs = run_analysis(
        full_rows,
        output_dir=output_dir,
        stem_suffix="",
        report_title="Genre-Based ALDi qualitative summary (full pooled test)",
    )
    control_outputs = run_analysis(
        gulf_control_rows,
        output_dir=output_dir,
        stem_suffix="_gulf_control",
        report_title="Genre-Based ALDi qualitative summary (Saudi/Gulf control test)",
    )

    print(f"[saved] {full_outputs['source']}")
    print(f"[saved] {full_outputs['genre']}")
    print(f"[saved] {full_outputs['duration_by_genre']}")
    print(f"[saved] {full_outputs['duration_by_dialect']}")
    print(f"[saved] {full_outputs['duration_by_dialect_and_genre']}")
    print(f"[saved] {full_outputs['boxplot']}")
    print(f"[saved] {full_outputs['mean_plot']}")
    print(f"[saved] {full_outputs['violin_plot']}")
    print(f"[saved] {full_outputs['duration_heatmap']}")
    print(f"[saved] {full_outputs['report']}")
    print(f"[saved] {control_outputs['source']}")
    print(f"[saved] {control_outputs['genre']}")
    print(f"[saved] {control_outputs['duration_by_genre']}")
    print(f"[saved] {control_outputs['duration_by_dialect']}")
    print(f"[saved] {control_outputs['duration_by_dialect_and_genre']}")
    print(f"[saved] {control_outputs['boxplot']}")
    print(f"[saved] {control_outputs['mean_plot']}")
    print(f"[saved] {control_outputs['violin_plot']}")
    print(f"[saved] {control_outputs['duration_heatmap']}")
    print(f"[saved] {control_outputs['report']}")


if __name__ == "__main__":
    main()
