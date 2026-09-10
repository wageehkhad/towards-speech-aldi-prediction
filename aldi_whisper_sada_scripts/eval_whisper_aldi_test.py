"""
Evaluate a trained Whisper-ALDi regressor on a held-out test manifest.

Example:
    python scripts/eval_whisper_aldi_test.py \
        --checkpoint checkpoints/whisper_aldi_sada/checkpoint_epoch4.pt \
        --test-manifest data/sada/manifest_test_aldi.csv \
        --batch-size 8 --num-workers 4 --device cuda

Outputs:
    - Prints test loss/MAE/Pearson/Spearman
    - Writes JSON to <checkpoint_dir>/test_metrics.json
"""

import argparse
import json
import os
import sys
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor

# Ensure we can import the training code
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
from train_whisper_aldi import (  # noqa: E402
    WhisperALDiRegressor,
    AudioALDiDataset,
    collate_batch,
    evaluate,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint_*.pt")
    ap.add_argument("--test-manifest", required=True, help="CSV with path, ALDi, etc.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument(
        "--local-only",
        action="store_true",
        help="Force transformers to use local cache only (HF_HUB_OFFLINE).",
    )
    ap.add_argument("--output", type=str, default=None, help="Optional path to write metrics JSON")
    ap.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print a progress line every N batches during eval (0 to disable).",
    )
    ap.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional: stop after this many batches (useful for quick sanity checks).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"[info] loading checkpoint {args.checkpoint}")
    load_start = time.time()
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"[ok] checkpoint loaded in {time.time()-load_start:.2f}s")

    ckpt_args = ckpt.get("args", {})
    model_size = ckpt_args.get("model_size", "medium")
    dataset_cache_dir = ckpt_args.get("dataset_cache_dir", "data/sada/hf_cache")
    freeze_encoder = ckpt_args.get("freeze_encoder", False)

    hf_kwargs = {"local_files_only": True} if args.local_only or os.environ.get("HF_HUB_OFFLINE") else {}
    print(f"[info] loading feature extractor openai/whisper-{model_size} (local_only={hf_kwargs.get('local_files_only', False)})")
    fe = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{model_size}", **hf_kwargs)

    print(f"[info] reading test manifest {args.test_manifest}")
    test_df = pd.read_csv(args.test_manifest)
    test_ds = AudioALDiDataset(
        test_df, fe, target_sr=fe.sampling_rate, dataset_cache_dir=dataset_cache_dir
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    model = WhisperALDiRegressor(
        model_id=f"openai/whisper-{model_size}", freeze_encoder=freeze_encoder
    ).to(device)
    print("[info] loading model weights")
    model.load_state_dict(ckpt["model_state"])

    print("[info] running evaluation...")
    eval_start = time.time()
    # Inline eval with progress logging
    model.eval()
    mse_loss = torch.nn.MSELoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            feats = batch["input_features"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            preds = model(feats, attention_mask=mask)
            loss = mse_loss(preds, labels)
            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if args.log_interval and (idx + 1) % args.log_interval == 0:
                print(f"[eval] batch {idx+1}/{len(test_loader)}")
            if args.max_batches is not None and (idx + 1) >= args.max_batches:
                print(f"[eval] stopping early at batch {idx+1} per --max-batches")
                break

    import numpy as np
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    mae = float(np.mean(np.abs(all_preds - all_labels)))
    from scipy.stats import pearsonr, spearmanr
    pearson = float(pearsonr(all_preds, all_labels)[0]) if len(all_preds) > 1 else float("nan")
    spearman = float(spearmanr(all_preds, all_labels)[0]) if len(all_preds) > 1 else float("nan")
    avg_loss = float(total_loss / max(1, len(test_loader)))
    print(f"[ok] evaluation finished in {time.time()-eval_start:.1f}s")
    metrics = {
        "test_loss": avg_loss,
        "test_mae": mae,
        "test_pearson": pearson,
        "test_spearman": spearman,
    }
    print(json.dumps(metrics, indent=2))

    out_path = args.output or os.path.join(os.path.dirname(args.checkpoint), "test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
