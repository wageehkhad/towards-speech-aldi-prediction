"""
Build a local Hugging Face dataset (train/validation/test) that bundles
SADA WAVs with ALDi scores. Requires that:
  - WAVs exist on disk (produced by download_sada.py with --audio-dir)
  - *_aldi.csv manifests exist (produced by score_sada_aldi.py)

Output: data/sada_aldi_hf/{train,validation,test}
You can load with `datasets.load_from_disk("data/sada_aldi_hf/train")`
or push to Hub manually if desired.
"""

import argparse
import os
from typing import Dict

import pandas as pd
from datasets import Audio, Dataset, DatasetDict

from submission_paths import ASSETS_DIR, SADA_DIR, path_str


def load_manifest(path: str, split: str) -> Dataset:
    df = pd.read_csv(path)
    if "path" not in df.columns or "ALDi" not in df.columns:
        raise ValueError(f"Manifest {path} must have 'path' and 'ALDi' columns.")
    if not df["path"].apply(os.path.isfile).all():
        missing = df[~df["path"].apply(os.path.isfile)]
        raise FileNotFoundError(
            f"{len(missing)} audio files missing; first few:\n{missing.head()}"
        )
    df = df.rename(columns={"path": "audio"})
    df["split"] = split
    cols = ["audio", "ALDi", "split"]
    if "text" in df.columns:
        cols.insert(1, "text")
    ds = Dataset.from_pandas(df[cols], preserve_index=False)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", default=path_str(SADA_DIR / "manifest_train_aldi.csv"))
    ap.add_argument("--val-manifest", default=path_str(SADA_DIR / "manifest_val_aldi.csv"))
    ap.add_argument("--test-manifest", default=path_str(SADA_DIR / "manifest_test_aldi.csv"))
    ap.add_argument("--out-dir", default=path_str(ASSETS_DIR / "sada_aldi_hf"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    splits: Dict[str, str] = {
        "train": args.train_manifest,
        "validation": args.val_manifest,
        "test": args.test_manifest,
    }

    dsd = DatasetDict(
        {name: load_manifest(path, name) for name, path in splits.items()}
    )

    for name, ds in dsd.items():
        out_split = os.path.join(args.out_dir, name)
        os.makedirs(out_split, exist_ok=True)
        ds.save_to_disk(out_split)
        print(f"[saved] {name} -> {out_split} ({len(ds)} rows)")

    print(
        "\nAll splits saved. Example load:\n"
        'from datasets import load_from_disk\n'
        'train = load_from_disk("data/sada_aldi_hf/train")\n'
        "print(train[0]['audio']['array'][:5], train[0]['ALDi'])"
    )


if __name__ == "__main__":
    main()
