"""
Attach Sentence-ALDi scores to the SADA manifest.
Input: data/sada/manifest.csv from download_sada.py
Output: data/sada/manifest_aldi.csv with ALDi score + bin.
"""

import argparse
import os
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from submission_paths import SADA_DIR, path_str


class ALDiScorer:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.mdl = self.mdl.to(dev).eval()
        self.device = dev

    def score(self, texts: List[str], batch_size: int = 32) -> List[float]:
        out: List[float] = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Scoring ALDi"):
                batch = texts[i : i + batch_size]
                enc = self.tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.mdl(**enc).logits
                vals = logits.squeeze(-1).detach().float().cpu().tolist()
                if isinstance(vals, float):
                    vals = [vals]
                out.extend(vals)
        return out


# Coarse three-way bucketing of the continuous score, used only for inspecting
# and stratifying the manifest. Training always uses the raw score.
def bin_score(x: float) -> str:
    if x < 0.33:
        return "low"
    if x < 0.66:
        return "mid"
    return "high"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=path_str(SADA_DIR / "manifest_train.csv"),
        help="Input manifest from download_sada.py",
    )
    ap.add_argument(
        "--output",
        default=path_str(SADA_DIR / "manifest_train_aldi.csv"),
        help="Output manifest with ALDi scores",
    )
    ap.add_argument(
        "--model",
        default="AMR-KELEG/Sentence-ALDi",
        help="HF model id for ALDi scoring",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--text-col",
        default="text",
        help="Column containing text in the manifest",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if args.text_col not in df.columns:
        raise ValueError(f"Manifest is missing text column '{args.text_col}'")
    texts = df[args.text_col].fillna("").astype(str).tolist()

    scorer = ALDiScorer(args.model, device=args.device)
    print(f"[info] Scoring {len(df):,} rows with {args.model} on {scorer.device}")
    scores = scorer.score(texts, batch_size=args.batch_size)
    df["ALDi"] = [round(float(s), 6) for s in scores]
    df["ALDi_bin"] = df["ALDi"].apply(bin_score)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[done] Wrote → {args.output}")
    print(
        df["ALDi"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string()
    )
    print("\n[histogram]")
    print(df["ALDi_bin"].value_counts(normalize=True).rename("fraction"))
    print("\n[next] Feed this manifest to train_whisper_aldi.py --target-hours 1.0")


if __name__ == "__main__":
    main()
