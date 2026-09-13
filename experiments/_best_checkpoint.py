"""
Print the path of the best-validation checkpoint in a training output directory.

Mirrors the selection error_analysis.py performs: highest val_pearson in
train_history.json, falling back to lowest val_loss. Used by the run_*.sh
scripts so the evaluation stage picks the same checkpoint the paper reports.
"""

import json
import os
import sys
from typing import Dict, List


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("usage: _best_checkpoint.py <checkpoint-dir>")

    ckpt_dir = sys.argv[1]
    history_path = os.path.join(ckpt_dir, "train_history.json")
    if not os.path.isfile(history_path):
        raise FileNotFoundError("Missing train history: {}".format(history_path))

    with open(history_path, "r", encoding="utf-8") as f:
        history: List[Dict[str, object]] = json.load(f)
    if not history:
        raise RuntimeError("Empty train history: {}".format(history_path))

    if "val_pearson" in history[0]:
        best = max(history, key=lambda x: x.get("val_pearson", float("-inf")))
    else:
        best = min(history, key=lambda x: x.get("val_loss", float("inf")))

    ckpt = os.path.join(ckpt_dir, "checkpoint_epoch{}.pt".format(int(best["epoch"])))
    if not os.path.isfile(ckpt):
        raise FileNotFoundError("Best checkpoint file not found: {}".format(ckpt))
    print(ckpt)


if __name__ == "__main__":
    main()
