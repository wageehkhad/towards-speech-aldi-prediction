"""
Per-dialect paired t-tests comparing the fine-tuned cascaded baseline (B2)
against the direct speech ALDi model on Casablanca.

Reads the same aligned per-utterance file as analyze_b2_vs_direct_wer.py, so
both scripts operate on identical rows.

Input columns:
    dataset          filtered to casablanca
    dialect_b2       dialect label
    true_aldi        silver-standard target
    pred_aldi_b2     cascaded fine-tuned prediction
    pred_aldi_direct direct model prediction

Output:
    paired_ttest_b2_vs_direct_by_dialect_rmse.csv
    paired_ttest_b2_vs_direct_by_dialect_rmse.txt

The test is a paired two-tailed t-test on per-utterance squared error, run as
ttest_rel(sq_err_b2, sq_err_direct). A positive t means the direct model has
the lower squared error. The ALL row pools every Casablanca utterance scored
by both systems.
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from submission_paths import RESULTS_DIR, path_str

REQUIRED_COLS = ["dataset", "dialect_b2", "true_aldi", "pred_aldi_b2", "pred_aldi_direct"]

COLUMNS = [
    "dialect",
    "n",
    "mse_b2",
    "mse_direct",
    "rmse_b2",
    "rmse_direct",
    "mean_se_diff_direct_minus_b2",
    "t_stat",
    "p_value",
    "significant_p_lt_0_05",
    "better_model_by_rmse",
]

HEADER = "Paired two-tailed t-test on per-utterance squared error (B2 vs Direct)"

OUTPUT_STEM = "paired_ttest_b2_vs_direct_by_dialect_rmse"


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_aligned(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Aligned predictions file not found: {}".format(csv_path))

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError("Aligned predictions missing columns in {}: {}".format(csv_path, missing))

    casa = df[df["dataset"] == "casablanca"].copy()
    casa = casa.dropna(subset=["true_aldi", "pred_aldi_b2", "pred_aldi_direct"])
    if casa.empty:
        raise RuntimeError("No Casablanca rows scored by both systems in {}".format(csv_path))

    casa["sq_err_b2"] = (casa["pred_aldi_b2"] - casa["true_aldi"]) ** 2
    casa["sq_err_direct"] = (casa["pred_aldi_direct"] - casa["true_aldi"]) ** 2
    print("[info] Casablanca utterances scored by both systems: {}".format(len(casa)))
    return casa


def test_group(label: str, grp: pd.DataFrame) -> Dict[str, object]:
    se_b2 = grp["sq_err_b2"].to_numpy(dtype=float)
    se_direct = grp["sq_err_direct"].to_numpy(dtype=float)

    mse_b2 = float(np.mean(se_b2))
    mse_direct = float(np.mean(se_direct))
    rmse_b2 = float(np.sqrt(mse_b2))
    rmse_direct = float(np.sqrt(mse_direct))

    if len(se_b2) > 1:
        result = ttest_rel(se_b2, se_direct)
        t_stat = float(result.statistic)
        p_value = float(result.pvalue)
    else:
        print("[warn] {} has {} paired rows, skipping the t-test".format(label, len(se_b2)))
        t_stat = float("nan")
        p_value = float("nan")

    return {
        "dialect": label,
        "n": int(len(se_b2)),
        "mse_b2": mse_b2,
        "mse_direct": mse_direct,
        "rmse_b2": rmse_b2,
        "rmse_direct": rmse_direct,
        "mean_se_diff_direct_minus_b2": float(np.mean(se_direct - se_b2)),
        "t_stat": t_stat,
        "p_value": p_value,
        "significant_p_lt_0_05": bool(p_value < 0.05) if p_value == p_value else False,
        "better_model_by_rmse": "Direct" if rmse_direct < rmse_b2 else "B2",
    }


def build_table(casa: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = [test_group("ALL", casa)]
    for dialect, grp in casa.groupby("dialect_b2"):
        rows.append(test_group(str(dialect), grp))
    return pd.DataFrame(rows, columns=COLUMNS)


def write_outputs(table: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)

    csv_path = os.path.join(out_dir, OUTPUT_STEM + ".csv")
    table.to_csv(csv_path, index=False)

    txt_path = os.path.join(out_dir, OUTPUT_STEM + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(HEADER + "\n")
        f.write("Columns: " + ", ".join(COLUMNS) + "\n\n")
        f.write(table.to_string(index=False) + "\n")

    print("[ok] wrote {}".format(csv_path))
    print("[ok] wrote {}".format(txt_path))


def verify(table: pd.DataFrame, reference_csv: str, tol: float) -> bool:
    if not os.path.exists(reference_csv):
        raise FileNotFoundError("Reference file not found: {}".format(reference_csv))

    ref = pd.read_csv(reference_csv).set_index("dialect")
    got = table.set_index("dialect")

    only_ref = sorted(set(ref.index) - set(got.index))
    only_got = sorted(set(got.index) - set(ref.index))
    if only_ref or only_got:
        print("[warn] row mismatch: missing={} unexpected={}".format(only_ref, only_got))
        return False

    numeric = [
        c for c in COLUMNS
        if c not in ("dialect", "significant_p_lt_0_05", "better_model_by_rmse")
    ]
    ok = True
    for dialect in ref.index:
        for col in numeric:
            expected = float(ref.loc[dialect, col])
            actual = float(got.loc[dialect, col])
            if not np.isclose(expected, actual, rtol=tol, atol=tol):
                print("[warn] {} {}: reference={!r} recomputed={!r}".format(dialect, col, expected, actual))
                ok = False
        for col in ("significant_p_lt_0_05", "better_model_by_rmse"):
            expected = str(ref.loc[dialect, col])
            actual = str(got.loc[dialect, col])
            if expected != actual:
                print("[warn] {} {}: reference={!r} recomputed={!r}".format(dialect, col, expected, actual))
                ok = False

    if ok:
        print("[ok] recomputed table matches {} to within {}".format(reference_csv, tol))
    return ok


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--aligned-csv",
        default=path_str(
            RESULTS_DIR
            / "baseline_comparison"
            / "aligned_b2_vs_direct_whisper_medium_full_all3_both_only.csv"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=path_str(RESULTS_DIR / "baseline_comparison"),
    )
    ap.add_argument(
        "--verify-against",
        default=None,
        help="Existing results CSV to compare against instead of writing output",
    )
    ap.add_argument("--tolerance", type=float, default=1e-9)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    casa = load_aligned(args.aligned_csv)
    table = build_table(casa)
    print(table.to_string(index=False))
    print()

    if args.verify_against:
        if not verify(table, args.verify_against, args.tolerance):
            raise RuntimeError("Recomputed table does not match {}".format(args.verify_against))
        return

    write_outputs(table, args.output_dir)


if __name__ == "__main__":
    main()
