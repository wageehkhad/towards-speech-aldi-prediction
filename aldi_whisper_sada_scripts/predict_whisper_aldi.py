"""
Predict ALDi scores for one or more wav files using a trained Whisper-ALDi checkpoint.

Example:
    python scripts/predict_whisper_aldi.py \
        --checkpoint checkpoints/whisper_aldi_sada_30h_medium/checkpoint_epoch3.pt \
        --wav path/to/audio1.wav path/to/audio2.wav \
        --device cuda
"""

import argparse
import json
import os
import sys
from typing import List

import torch
import torchaudio
from transformers import WhisperFeatureExtractor

# allow importing the model definition
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
from train_whisper_aldi import WhisperALDiRegressor  # noqa: E402


def load_wav(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.dim() > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint_epoch*.pt")
    ap.add_argument("--wav", nargs="+", required=True, help="One or more wav files")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument(
        "--local-only",
        action="store_true",
        help="Force transformers to use local cache only (HF_HUB_OFFLINE).",
    )
    ap.add_argument("--output-jsonl", type=str, default=None, help="Optional output path for predictions")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model_size = ckpt_args.get("model_size", "medium")
    freeze_encoder = ckpt_args.get("freeze_encoder", False)

    hf_kwargs = {"local_files_only": True} if args.local_only or os.environ.get("HF_HUB_OFFLINE") else {}
    fe = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{model_size}", **hf_kwargs)
    target_sr = fe.sampling_rate

    model = WhisperALDiRegressor(
        model_id=f"openai/whisper-{model_size}", freeze_encoder=freeze_encoder
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    results = []
    with torch.no_grad():
        for path in args.wav:
            if not os.path.isfile(path):
                print(f"[warn] skipping missing file: {path}", file=sys.stderr)
                continue
            wav = load_wav(path, target_sr=target_sr)
            feats = fe(wav.numpy(), sampling_rate=target_sr, return_tensors="pt")
            input_feats = feats["input_features"].to(device)  # (1, 80, T)
            if "attention_mask" in feats:
                attn = feats["attention_mask"].to(device)
            else:
                # Whisper extractor sometimes omits mask; create full-ones mask over time dimension
                attn = torch.ones(input_feats.shape[-1], device=device, dtype=torch.long).unsqueeze(0)
            preds = model(input_feats, attention_mask=attn)
            score = float(preds.squeeze().cpu().item())
            results.append({"path": path, "aldi": score})
            print(json.dumps(results[-1]))

    if args.output_jsonl:
        with open(args.output_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"[saved] {args.output_jsonl}")


if __name__ == "__main__":
    main()
