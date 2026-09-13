"""
Reconstruct the full list of genre-validation clips.

Combines the source windows in assets/genre_based_aldi/sources.csv with the
per-source clip counts in results/genre_based_aldi/summary_by_source.csv to
produce one row per clip, with the absolute time span each clip occupies in
its original video.

Segmentation is deterministic (fixed-length windows from the start of the
trimmed audio, see segment_audio.py), so clip N of a source always covers the
same span. That makes the clip set reproducible from metadata alone, without
redistributing the audio.

Output:
    assets/genre_based_aldi/clip_manifest.csv
"""

import argparse
import csv
import os
from typing import Dict, List

DEFAULT_CLIP_SECONDS = 15

COLUMNS = [
    "clip_id",
    "source_id",
    "clip_index",
    "genre",
    "dialect_region",
    "url",
    "window_start",
    "window_end",
    "clip_start_in_video",
    "clip_end_in_video",
]


def parse_hms(value: str) -> int:
    # sources.csv is HH:MM:SS, but one row carries a non-normalised value
    # (00:00:60), so seconds and minutes are summed rather than validated.
    parts = [int(p) for p in value.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts[0], parts[1], parts[2]
    return hours * 3600 + minutes * 60 + seconds


def format_hms(total: int) -> str:
    return "{:02d}:{:02d}:{:02d}".format(total // 3600, (total % 3600) // 60, total % 60)


def load_counts(path: str) -> Dict[str, int]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["source_id"]: int(row["n_clips"]) for row in csv.DictReader(f)}


def build_rows(sources_csv: str, counts: Dict[str, int], clip_seconds: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    missing: List[str] = []

    with open(sources_csv, newline="", encoding="utf-8") as f:
        for source in csv.DictReader(f):
            source_id = os.path.splitext(source["filename"])[0]
            if source_id not in counts:
                missing.append(source_id)
                continue

            window_start = parse_hms(source["start_time"])
            for index in range(counts[source_id]):
                offset = window_start + index * clip_seconds
                rows.append(
                    {
                        "clip_id": "{}__clip{:04d}.wav".format(source_id, index),
                        "source_id": source_id,
                        "clip_index": index,
                        "genre": source["genre"],
                        "dialect_region": source["dialect_region"],
                        "url": source["url"],
                        "window_start": source["start_time"],
                        "window_end": source["end_time"],
                        "clip_start_in_video": format_hms(offset),
                        "clip_end_in_video": format_hms(offset + clip_seconds),
                    }
                )

    if missing:
        print("[warn] no clip count for: {}".format(", ".join(missing)))
    return rows


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    ap = argparse.ArgumentParser()
    ap.add_argument("--sources-csv", default=os.path.join(root, "assets", "genre_based_aldi", "sources.csv"))
    ap.add_argument("--summary-csv", default=os.path.join(root, "results", "genre_based_aldi", "summary_by_source.csv"))
    ap.add_argument("--output", default=os.path.join(root, "assets", "genre_based_aldi", "clip_manifest.csv"))
    ap.add_argument("--clip-seconds", type=int, default=DEFAULT_CLIP_SECONDS)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    counts = load_counts(args.summary_csv)
    rows = build_rows(args.sources_csv, counts, args.clip_seconds)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("[ok] wrote {} clips from {} sources to {}".format(len(rows), len(counts), args.output))


if __name__ == "__main__":
    main()
