"""
Download SADA22 from Hugging Face, inspect columns, and export a manifest with
audio paths, durations, and transcripts. Keeps everything under data/sada.
"""

import argparse
import csv
import os
from typing import Optional, Tuple, List

from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm

from submission_paths import HF_CACHE_DIR, SADA_DIR, path_str


def infer_columns(column_names: List[str]) -> Tuple[str, str]:
    audio_col = None
    text_col = None
    for name in ["audio", "Audio", "speech"]:
        if name in column_names:
            audio_col = name
            break
    for name in ["text", "transcription", "sentence", "transcript", "normalized_text"]:
        if name in column_names:
            text_col = name
            break
    if audio_col is None or text_col is None:
        raise ValueError(f"Could not infer columns. Found: {column_names}")
    return audio_col, text_col


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="MohamedRashad/SADA22",
        help="HF dataset id",
    )
    ap.add_argument(
        "--split",
        default="train",
        help="Split to load (use 'train+validation+test' to merge if available)",
    )
    ap.add_argument(
        "--cache-dir",
        default=path_str(HF_CACHE_DIR),
        help="Where to cache the downloaded dataset",
    )
    ap.add_argument(
        "--manifest",
        default=path_str(SADA_DIR / "manifest_train.csv"),
        help="Where to write the manifest CSV",
    )
    ap.add_argument(
        "--audio-dir",
        default=path_str(SADA_DIR / "wavs"),
        help="Where to store extracted WAV files (split subfolder will be created)",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Parallel writers for saving WAV files",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick smoke runs",
    )
    args = ap.parse_args()

    split_dir = os.path.join(args.audio_dir, args.split)
    ensure_dir(split_dir)
    ensure_dir(os.path.dirname(args.manifest))
    ds = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    audio_col, text_col = infer_columns(ds.column_names)
    print(f"[info] Loaded split={args.split} with {len(ds):,} rows")
    print(f"[info] Using columns → audio='{audio_col}', text='{text_col}'")

    # Infer sample rate from feature metadata when possible
    sr_meta: Optional[int] = None
    try:
        sr_meta = ds.features[audio_col].sampling_rate  # type: ignore[attr-defined]
    except Exception:
        pass

    total_sec = 0.0
    min_sec, max_sec = 1e9, 0.0
    n_rows = 0

    iterable = ds if args.limit is None else ds.select(range(args.limit))

    def process(idx_row):
        idx, row = idx_row
        audio = row[audio_col]
        text = row[text_col]
        array = audio.get("array") if isinstance(audio, dict) else None
        sr = audio.get("sampling_rate") if isinstance(audio, dict) else sr_meta
        out_path = audio.get("path") if isinstance(audio, dict) else None
        if out_path is None or not os.path.isfile(out_path):
            fname = f"{row.get('id', idx)}.wav"
            out_path = os.path.join(split_dir, fname)
            if array is None:
                raise ValueError("Audio array missing and no valid path to reuse.")
            sf.write(out_path, array, int(sr or sr_meta or 16000))

        if array is not None and sr:
            duration = len(array) / sr
        elif isinstance(audio, dict) and "duration" in audio:
            duration = float(audio["duration"])
        else:
            duration = None

        return (
            row.get("id", idx),
            args.split,
            out_path,
            f"{duration:.3f}" if duration is not None else "",
            text.strip() if isinstance(text, str) else "",
            sr or "",
            duration,
        )

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "split", "path", "duration_sec", "text", "sampling_rate_hint"]
        )
        total_items = len(iterable)
        pending = set()
        iterator = enumerate(iterable)
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex, tqdm(total=total_items, desc="Building manifest") as pbar:
            try:
                while True:
                    while len(pending) < args.num_workers * 2:
                        try:
                            idx_row = next(iterator)
                        except StopIteration:
                            break
                        pending.add(ex.submit(process, idx_row))

                    if not pending:
                        break

                    for fut in as_completed(pending):
                        pending.remove(fut)
                        rid, split, out_path, dur_str, text_str, sr_hint, duration = fut.result()
                        writer.writerow([rid, split, out_path, dur_str, text_str, sr_hint])
                        if duration is not None:
                            total_sec += duration
                            min_sec = min(min_sec, duration)
                            max_sec = max(max_sec, duration)
                        n_rows += 1
                        pbar.update(1)
                        break  # step back to refill queue and refresh pbar
            except KeyboardInterrupt:
                for fut in pending:
                    fut.cancel()
                raise

    hrs = total_sec / 3600
    print(f"[done] Wrote manifest → {args.manifest}")
    print(
        f"[stats] rows={n_rows:,} | hours≈{hrs:.1f} | dur_min={min_sec:.2f}s | dur_max={max_sec:.2f}s | sr_hint={sr_meta}"
    )
    print(
        "[next] Run score_sada_aldi.py to attach ALDi labels, "
        "then train_whisper_aldi.py with --target-hours 1.0"
    )


if __name__ == "__main__":
    main()
