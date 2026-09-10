#!/usr/bin/env python3
"""
Score every genre-validation clip with a direct Whisper ALDi checkpoint.

Writes clip-level predictions plus a run manifest recording the checkpoint and
settings the run used. Model size and encoder-freezing are read back from the
args stored inside the checkpoint rather than passed in, so the architecture
always matches the one that was trained.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
import torchaudio
from torch import nn
from transformers import WhisperFeatureExtractor, WhisperModel


CLIP_RE = re.compile(r"^(?P<source_id>.+)__clip(?P<clip_idx>\d+)\.wav$")


class WhisperALDiRegressor(nn.Module):
    def __init__(self, model_id: str, freeze_encoder: bool = False, local_files_only: bool = True):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        ).encoder
        hidden = self.encoder.config.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        enc_outputs = self.encoder(input_features=input_features, attention_mask=attention_mask)
        enc_out = enc_outputs.last_hidden_state
        reduced_mask = torch.nn.functional.interpolate(
            attention_mask.unsqueeze(1).float(),
            size=enc_out.size(1),
            mode="nearest",
        ).squeeze(1)
        mask = reduced_mask.unsqueeze(-1)
        pooled = (enc_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def load_wav(path: Path, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.dim() > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    assets_dir = root / "assets" / "genre_based_aldi"
    results_dir = root / "results" / "genre_based_aldi"
    ap = argparse.ArgumentParser(description="Run direct Whisper-ALDi inference on genre-validation clips.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--clips-dir", default=str(assets_dir / "clips"))
    ap.add_argument("--metadata", default=str(assets_dir / "sources.csv"))
    ap.add_argument("--output-csv", default=str(results_dir / "clip_predictions.csv"))
    ap.add_argument("--output-json", default=str(results_dir / "run_manifest.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    return ap.parse_args()


def read_metadata(metadata_path: Path):
    by_source = {}
    with metadata_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_source[Path(row["filename"]).stem] = row
    return by_source


def main():
    args = parse_args()
    clips_dir = Path(args.clips_dir)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    clip_paths = sorted(clips_dir.glob("*.wav"))
    if not clip_paths:
        raise RuntimeError(f"No clips found in {clips_dir}")

    metadata = read_metadata(Path(args.metadata))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model_size = ckpt_args.get("model_size", "medium")
    freeze_encoder = ckpt_args.get("freeze_encoder", False)
    model_id = f"openai/whisper-{model_size}"

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id, local_files_only=True)
    model = WhisperALDiRegressor(
        model_id=model_id,
        freeze_encoder=freeze_encoder,
        local_files_only=True,
    )
    model.load_state_dict(ckpt["model_state"])
    if device.type == "cuda":
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    else:
        model = model.to(device)
    model.eval()

    fieldnames = [
        "clip_path",
        "source_id",
        "clip_index",
        "genre",
        "subgenre",
        "source_label",
        "dialect_region",
        "expected_aldi",
        "predicted_aldi",
        "duration_sec",
    ]
    n_rows = 0
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for start in range(0, len(clip_paths), args.batch_size):
                batch_paths = clip_paths[start : start + args.batch_size]
                batch_meta = []
                batch_wavs = []
                for clip_path in batch_paths:
                    match = CLIP_RE.match(clip_path.name)
                    if not match:
                        print(f"[warn] skipping unexpected clip name: {clip_path.name}")
                        continue
                    source_id = match.group("source_id")
                    clip_index = int(match.group("clip_idx"))
                    row_meta = metadata.get(source_id, {})
                    wav = load_wav(clip_path, target_sr=feature_extractor.sampling_rate)
                    batch_wavs.append(wav.numpy())
                    batch_meta.append((clip_path, source_id, clip_index, row_meta, wav))

                if not batch_meta:
                    continue

                feats = feature_extractor(
                    batch_wavs,
                    sampling_rate=feature_extractor.sampling_rate,
                    return_tensors="pt",
                    padding="max_length",
                )
                input_feats = feats["input_features"].to(device)
                # Match the repo's reference predictor exactly: use a full-ones
                # frame mask over the extracted mel features.
                attention_mask = torch.ones(
                    (input_feats.shape[0], input_feats.shape[-1]),
                    device=device,
                    dtype=torch.long,
                )
                preds = model(input_feats, attention_mask=attention_mask).detach().cpu().tolist()
                if isinstance(preds, float):
                    preds = [preds]

                for pred_value, (clip_path, source_id, clip_index, row_meta, wav) in zip(preds, batch_meta):
                    duration_sec = round(float(wav.numel() / feature_extractor.sampling_rate), 3)
                    out_row = {
                        "clip_path": str(clip_path),
                        "source_id": source_id,
                        "clip_index": clip_index,
                        "genre": row_meta.get("genre", ""),
                        "subgenre": row_meta.get("subgenre", ""),
                        "source_label": row_meta.get("source_label", ""),
                        "dialect_region": row_meta.get("dialect_region", ""),
                        "expected_aldi": row_meta.get("expected_aldi", ""),
                        "predicted_aldi": round(float(pred_value), 6),
                        "duration_sec": duration_sec,
                    }
                    writer.writerow(out_row)
                    n_rows += 1
                    print(json.dumps(out_row, ensure_ascii=False))

    run_manifest = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": str(device),
        "model_size": model_size,
        "freeze_encoder": bool(freeze_encoder),
        "batch_size": args.batch_size,
        "clips_dir": str(clips_dir.resolve()),
        "n_predictions": n_rows,
        "output_csv": str(output_csv.resolve()),
    }
    output_json.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"[saved] {output_csv}")
    print(f"[saved] {output_json}")


if __name__ == "__main__":
    main()
