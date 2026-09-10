#!/usr/bin/env python3
"""
Cut each trimmed source WAV into fixed-length clips for ALDi scoring.

Input is assets/genre_based_aldi/raw_audio, output is assets/genre_based_aldi/clips.
Windows are fixed-length and not aligned to sentence boundaries. Files here are
already trimmed to the window chosen in sources.csv, so no further offset is
applied. The trailing remainder of each file is dropped when it is shorter than
--min-clip-ms.
"""

import argparse
from pathlib import Path

from pydub import AudioSegment


def parse_args():
    root = Path(__file__).resolve().parents[1]
    assets_dir = root / "assets" / "genre_based_aldi"
    ap = argparse.ArgumentParser(description="Segment trimmed genre-validation WAVs into fixed clips.")
    ap.add_argument("--input-dir", default=str(assets_dir / "raw_audio"))
    ap.add_argument("--output-dir", default=str(assets_dir / "clips"))
    ap.add_argument("--clip-length-ms", type=int, default=15000)
    ap.add_argument("--min-clip-ms", type=int, default=5000)
    return ap.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_paths = sorted(input_dir.glob("*.wav"))
    if not wav_paths:
        raise RuntimeError(f"No WAV files found in {input_dir}")

    total = 0
    for wav_path in wav_paths:
        source_id = wav_path.stem
        audio = AudioSegment.from_wav(wav_path)
        print(f"[segment] {wav_path.name} duration_sec={len(audio) / 1000:.1f}")
        for i, start in enumerate(range(0, len(audio), args.clip_length_ms)):
            clip = audio[start : start + args.clip_length_ms]
            if len(clip) < args.min_clip_ms:
                continue
            out_name = f"{source_id}__clip{i:04d}.wav"
            clip.export(output_dir / out_name, format="wav")
            total += 1
    print(f"[done] wrote {total} clips to {output_dir}")


if __name__ == "__main__":
    main()
