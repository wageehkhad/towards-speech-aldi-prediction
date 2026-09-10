#!/usr/bin/env python3
"""
Download the trimmed audio window for each genre-validation source.

Reads assets/genre_based_aldi/sources.csv and fetches only the section between
start_time and end_time for each URL, converted to 16 kHz mono WAV to match the
sample rate the ALDi models expect. Output goes to assets/genre_based_aldi/raw_audio.

Sources are windowed rather than downloaded whole so that intros and outros are
excluded before segmentation.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    root = Path(__file__).resolve().parents[1]
    assets_dir = root / "assets" / "genre_based_aldi"
    ap = argparse.ArgumentParser(description="Download trimmed YouTube audio as 16 kHz mono WAV.")
    ap.add_argument("--metadata", default=str(assets_dir / "sources.csv"))
    ap.add_argument("--output-dir", default=str(assets_dir / "raw_audio"))
    ap.add_argument("--yt-dlp-bin", default=None, help="Optional explicit yt-dlp executable path")
    ap.add_argument("--cookies", default=None, help="Optional Netscape-format cookies.txt path")
    ap.add_argument("--js-runtime", default=None, help="Optional JS runtime spec for yt-dlp, e.g. deno:/path/to/deno")
    ap.add_argument("--force", action="store_true", help="Redownload files even if they already exist")
    return ap.parse_args()


def find_ytdlp(explicit: str = None) -> list:
    if explicit:
        return [explicit]
    standalone = Path("/tmp/yt-dlp_linux")
    if standalone.exists() and os.access(standalone, os.X_OK):
        return [str(standalone)]
    binary = shutil.which("yt-dlp")
    if binary:
        return [binary]
    try:
        import yt_dlp  # noqa: F401

        return [sys.executable, "-m", "yt_dlp"]
    except Exception as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Install it in the project venv with "
            "`python -m pip install yt-dlp`."
        ) from exc


def read_rows(metadata_path: Path):
    with metadata_path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def main():
    args = parse_args()
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ytdlp_cmd = find_ytdlp(args.yt_dlp_bin)
    rows = list(read_rows(metadata_path))
    if not rows:
        raise RuntimeError(f"No rows found in metadata file: {metadata_path}")

    failures = []
    for row in rows:
        filename = row["filename"]
        url = row["url"]
        start_time = row["start_time"]
        end_time = row["end_time"]
        out_path = output_dir / filename

        if out_path.exists() and not args.force:
            print(f"[skip] {out_path}")
            continue

        out_template = str(out_path.with_suffix(".%(ext)s"))
        cmd = [
            *ytdlp_cmd,
            "--force-overwrites",
            "--no-playlist",
            "-f",
            "bestaudio/best",
            "-x",
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",
            "--postprocessor-args",
            "ffmpeg:-ar 16000 -ac 1",
            "--download-sections",
            f"*{start_time}-{end_time}",
            "-o",
            out_template,
            url,
        ]
        if args.cookies:
            cmd[1:1] = ["--cookies", args.cookies]
        if args.js_runtime:
            cmd[1:1] = ["--js-runtimes", args.js_runtime]
        print(f"[download] {filename} <- {url} [{start_time} to {end_time}]")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append({"filename": filename, "url": url, "returncode": exc.returncode})
            print(f"[error] failed: {filename} ({url})")
            continue

        if not out_path.exists():
            matches = sorted(output_dir.glob(f"{Path(filename).stem}.*"))
            if not matches:
                failures.append({"filename": filename, "url": url, "returncode": "missing_output"})
                print(f"[error] missing output file for {filename}")
                continue
            if matches[0] != out_path:
                matches[0].rename(out_path)
        print(f"[saved] {out_path}")

    if failures:
        print("[summary] some downloads failed:")
        for item in failures:
            print(f"  - {item['filename']}: {item['url']} ({item['returncode']})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
