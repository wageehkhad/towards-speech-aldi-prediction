"""
Build a per-dialect qualitative sample table for Casablanca.

For a sample of utterances per dialect, produces the gold transcript, the
transcript from the cascaded ASR model, the word error rate between them, and
three ALDi scores: the target taken from the gold transcript, the cascaded
estimate taken from the ASR transcript, and the direct model's estimate taken
from the audio. The point is to show by example where transcription error
pulls the cascaded estimate away from the target while the direct estimate,
which never sees a transcript, stays closer.

Selection runs in two passes so the final table spans a range of error rates.
Pass one takes a pool per dialect spread across utterance duration. Pass two
transcribes the pool, then keeps the utterances sitting at evenly spaced WER
quantiles, so each dialect contributes low, middle and high error examples.

WER and the ALDi scorer are imported from the modules the reported experiments
use, so the numbers here are computed the same way as the ones in the paper.

Output:
    results/casablanca_qualitative_samples.xlsx
    results/casablanca_qualitative_samples.csv
    results/casablanca_qualitative_pool.csv   (the full transcribed pool)
"""

import argparse
import gc
import os
import sys
from typing import Dict, List, Optional

import pandas as pd
import torch
import torchaudio
from transformers import WhisperFeatureExtractor

from analyze_b2_vs_direct_wer import word_error_rate
from asr_aldi_baseline_utils import SentenceALDiScorer, WhisperASRTranscriber
from submission_paths import CASABLANCA_DIR, MODELS_DIR, RESULTS_DIR, path_str

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from train_whisper_aldi import WhisperALDiRegressor  # noqa: E402

MIN_WORDS = 3
MIN_DURATION_SEC = 0.5

COLUMNS = [
    "sample_id",
    "dialect",
    "n_tokens_gold",
    "duration_sec",
    "gold_transcript",
    "cascaded_transcript",
    "wer",
    "aldi_gold",
    "aldi_cascaded",
    "aldi_direct",
    "cascaded_minus_gold",
    "direct_minus_gold",
]


def is_usable(text: str) -> bool:
    stripped = text.strip()
    if len(stripped.split()) < MIN_WORDS:
        return False
    # Utterances that are only an annotation marker such as [music] give a
    # meaningless WER, so they are excluded from the pool.
    if stripped.startswith("[") and stripped.endswith("]"):
        return False
    return True


def free_memory() -> None:
    # Call after the caller has dropped its own reference to a model. Each
    # Whisper-medium is roughly 3 GB in fp32 and the stages run one at a time,
    # so releasing each before the next loads keeps the peak at a single model.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_direct_checkpoint(model_arg: str, filename: str) -> str:
    # Accepts either a local checkpoint_*.pt or a Hugging Face repo id, so the
    # released model can be used without downloading it by hand first.
    if os.path.isfile(model_arg):
        return model_arg

    from huggingface_hub import hf_hub_download

    print("[info] fetching {} from the hub repo {}".format(filename, model_arg))
    return hf_hub_download(repo_id=model_arg, filename=filename)


def load_direct_model(checkpoint_path: str, device: torch.device, local_only: bool):
    # Mirrors predict_whisper_aldi.py: the encoder size and freezing flag are
    # read back from the args stored in the checkpoint, so the architecture
    # always matches the one that was trained.
    # weights_only=False is required: the checkpoint stores the training args
    # dict alongside the tensors, and torch >= 2.6 defaults the flag to True.
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # torch < 2.0 has no weights_only parameter.
        ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model_size = ckpt_args.get("model_size", "medium")
    freeze_encoder = ckpt_args.get("freeze_encoder", False)

    hf_kwargs = {"local_files_only": True} if local_only else {}
    model_id = "openai/whisper-{}".format(model_size)
    extractor = WhisperFeatureExtractor.from_pretrained(model_id, **hf_kwargs)

    model = WhisperALDiRegressor(model_id=model_id, freeze_encoder=freeze_encoder).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("[info] direct model: whisper-{}, frozen encoder={}".format(model_size, freeze_encoder))
    return model, extractor


def score_direct(samples: List[object], model, extractor, device: torch.device, batch_size: int) -> List[float]:
    target_sr = extractor.sampling_rate
    scores: List[float] = []

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            waves = []
            for sample in batch:
                audio = sample.audio_input
                if not isinstance(audio, dict) or "array" not in audio:
                    raise RuntimeError(
                        "Audio for {} is missing its samples; something consumed it "
                        "before the direct model ran".format(sample.sample_id)
                    )
                wav = torch.tensor(audio["array"], dtype=torch.float32)
                if wav.dim() > 1:
                    wav = wav.mean(dim=0)
                source_sr = int(audio["sampling_rate"])
                if source_sr != target_sr:
                    wav = torchaudio.functional.resample(wav.unsqueeze(0), source_sr, target_sr).squeeze(0)
                waves.append(wav.numpy())

            feats = extractor(waves, sampling_rate=target_sr, return_tensors="pt")
            input_features = feats["input_features"].to(device)
            if "attention_mask" in feats:
                attention_mask = feats["attention_mask"].to(device)
            else:
                # The Whisper extractor omits the mask; the reference predictor
                # uses a full-ones mask over the mel frames, so match that.
                attention_mask = torch.ones(
                    input_features.shape[0], input_features.shape[-1], device=device, dtype=torch.long
                )

            preds = model(input_features, attention_mask=attention_mask)
            scores.extend(float(x) for x in preds.detach().squeeze(-1).cpu().tolist())
            print("[info]   {}/{}".format(min(start + batch_size, len(samples)), len(samples)))

    return scores


class Utterance(object):
    """Metadata for one utterance; audio is attached later for the pool only."""

    __slots__ = ("sample_id", "dialect", "reference_text", "duration_sec", "true_aldi", "audio_input")

    def __init__(self, sample_id, dialect, reference_text, duration_sec, true_aldi):
        self.sample_id = sample_id
        self.dialect = dialect
        self.reference_text = reference_text
        self.duration_sec = duration_sec
        self.true_aldi = true_aldi
        self.audio_input = None


def load_metadata(metadata_csv: str) -> List[Utterance]:
    # Selection needs only the scored CSV. Decoding audio for all 6,819
    # utterances just to choose 80 of them costs about 2 GB of memory, which is
    # enough to get the process killed on a modest machine.
    meta = pd.read_csv(metadata_csv)
    utterances = [
        Utterance(
            sample_id=str(row["id"]),
            dialect=str(row["dialect"]),
            reference_text=str(row["text"]),
            duration_sec=float(row["duration"]),
            true_aldi=float(row["aldi_score"]),
        )
        for row in meta.to_dict(orient="records")
    ]
    print("[info] metadata for {} utterances".format(len(utterances)))
    return utterances


def attach_audio(pool: List[Utterance], hf_dataset_dir: str, split: str) -> List[Utterance]:
    from datasets import load_from_disk

    dataset = load_from_disk(hf_dataset_dir)
    if not hasattr(dataset, "column_names") or isinstance(dataset.column_names, dict):
        dataset = dataset[split]

    id_column = "seg_id" if "seg_id" in dataset.column_names else "id"
    # Reading one column does not decode audio, so this index is cheap.
    positions = {str(v): i for i, v in enumerate(dataset[id_column])}

    attached: List[Utterance] = []
    missing: List[str] = []
    for utterance in pool:
        index = positions.get(utterance.sample_id)
        if index is None:
            missing.append(utterance.sample_id)
            continue
        audio = dataset[index]["audio"]
        utterance.audio_input = {
            "array": audio["array"],
            "sampling_rate": int(audio["sampling_rate"]),
        }
        attached.append(utterance)

    if missing:
        print("[warn] no audio found for {} utterance(s), first few: {}".format(len(missing), missing[:5]))
    print("[info] audio attached for {} utterances".format(len(attached)))
    return attached


def select_pool(samples: List[object], per_dialect: int, seed: int) -> List[object]:
    by_dialect: Dict[str, List[object]] = {}
    for sample in samples:
        # Only zero-length clips are dropped. The reported evaluation scores the
        # full Casablanca validation split with no duration filtering, so the
        # sample stays faithful to it: clips Whisper cannot transcribe are part
        # of what the cascaded approach is measured on.
        if sample.duration_sec < MIN_DURATION_SEC:
            continue
        if is_usable(sample.reference_text):
            by_dialect.setdefault(sample.dialect, []).append(sample)

    pool: List[object] = []
    for dialect in sorted(by_dialect):
        group = sorted(by_dialect[dialect], key=lambda s: len(s.reference_text.split()))
        if len(group) <= per_dialect:
            pool.extend(group)
            continue
        # Even positions across the group sorted by transcript token count, so
        # the pool spans short and long utterances rather than clustering at the
        # median. Token count rather than duration, because that is what governs
        # how much lexical evidence of dialectness a transcript carries.
        step = (len(group) - 1) / float(per_dialect - 1)
        picked = {int(round(i * step)) for i in range(per_dialect)}
        pool.extend(group[i] for i in sorted(picked))

    print("[info] pool: {} utterances across {} dialects".format(len(pool), len(by_dialect)))
    return pool


def select_final(pool_df: pd.DataFrame, per_dialect: int) -> pd.DataFrame:
    keep = []
    for dialect, group in pool_df.groupby("dialect"):
        group = group.sort_values("wer").reset_index(drop=True)
        if len(group) <= per_dialect:
            keep.append(group)
            continue
        step = (len(group) - 1) / float(per_dialect - 1)
        idx = sorted({int(round(i * step)) for i in range(per_dialect)})
        keep.append(group.iloc[idx])
    out = pd.concat(keep, ignore_index=True)
    return out.sort_values(["dialect", "wer"]).reset_index(drop=True)


def write_outputs(final_df: pd.DataFrame, pool_df: pd.DataFrame, out_dir: str) -> None:
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    pool_path = os.path.join(out_dir, "casablanca_qualitative_pool.csv")
    pool_df.to_csv(pool_path, index=False)
    print("[ok] wrote {}".format(pool_path))

    csv_path = os.path.join(out_dir, "casablanca_qualitative_samples.csv")
    final_df.to_csv(csv_path, index=False)
    print("[ok] wrote {}".format(csv_path))

    xlsx_path = os.path.join(out_dir, "casablanca_qualitative_samples.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            final_df.to_excel(writer, sheet_name="samples", index=False)
            sheet = writer.sheets["samples"]
            widths = {
                "A": 12, "B": 12, "C": 10, "D": 11, "E": 60, "F": 60,
                "G": 8, "H": 11, "I": 13, "J": 11, "K": 18, "L": 17,
            }
            for column, width in widths.items():
                sheet.column_dimensions[column].width = width
            sheet.freeze_panes = "A2"
        print("[ok] wrote {}".format(xlsx_path))
    except ImportError:
        print("[warn] openpyxl not installed, skipped the xlsx (the csv above has the same rows)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dataset", default=path_str(CASABLANCA_DIR / "casablanca_audio_aldi"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--metadata-csv", default=path_str(CASABLANCA_DIR / "casablanca_aldi_scored.csv"))
    ap.add_argument("--asr-model", default=path_str(MODELS_DIR / "whisper_sada_asr_full"))
    ap.add_argument("--aldi-model", default="AMR-KELEG/Sentence-ALDi")
    ap.add_argument(
        "--direct-model",
        default=path_str(MODELS_DIR / "whisper_aldi_sada_full_medium" / "checkpoint_epoch5.pt"),
        help="Local checkpoint_*.pt, or a Hugging Face repo id holding one",
    )
    ap.add_argument(
        "--direct-checkpoint-filename",
        default="checkpoint_epoch5.pt",
        help="File to pull when --direct-model names a hub repo",
    )
    ap.add_argument("--direct-batch-size", type=int, default=4)
    ap.add_argument("--pool-per-dialect", type=int, default=20)
    ap.add_argument("--samples-per-dialect", type=int, default=5)
    ap.add_argument("--asr-batch-size", type=int, default=4)
    ap.add_argument(
        "--chunk-length-s",
        type=float,
        default=0.0,
        help="Whisper chunking window; 0 disables it. Casablanca clips are well "
             "under 30s, and padding them out to a chunk lets Whisper hallucinate "
             "repetition loops into the silence.",
    )
    ap.add_argument("--aldi-batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--output-dir", default=path_str(RESULTS_DIR))
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print("[stage] reading the scored metadata")
    utterances = load_metadata(args.metadata_csv)

    pool = select_pool(utterances, args.pool_per_dialect, args.seed)

    print("[stage] attaching audio for the pool")
    pool = attach_audio(pool, args.hf_dataset, args.split)
    if not pool:
        raise RuntimeError("No pool utterance could be matched to audio in {}".format(args.hf_dataset))

    print("[stage] transcribing the pool with {}".format(args.asr_model))
    transcriber = WhisperASRTranscriber(
        asr_model=args.asr_model,
        device=args.device,
        local_only=args.local_only,
        chunk_length_s=args.chunk_length_s if args.chunk_length_s > 0 else None,
    )
    transcripts: List[str] = []
    for start in range(0, len(pool), args.asr_batch_size):
        batch = pool[start : start + args.asr_batch_size]
        # The ASR pipeline pops "array" out of the dict it is handed, so pass
        # shallow copies and keep each utterance's own audio intact for the
        # direct model, which runs over the same clips afterwards.
        audio_copies = [dict(s.audio_input) for s in batch]
        rows = transcriber.transcribe_batch(audio_copies, batch_size=args.asr_batch_size)
        transcripts.extend(r["text"] for r in rows)
        print("[info]   {}/{}".format(min(start + args.asr_batch_size, len(pool)), len(pool)))

    del transcriber
    free_memory()

    print("[stage] scoring the audio with the direct model")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    direct_ckpt = resolve_direct_checkpoint(args.direct_model, args.direct_checkpoint_filename)
    direct_model, extractor = load_direct_model(direct_ckpt, device, args.local_only)
    aldi_direct = score_direct(pool, direct_model, extractor, device, args.direct_batch_size)
    del direct_model, extractor
    free_memory()

    print("[stage] scoring the cascaded transcripts with Sentence-ALDi")
    scorer = SentenceALDiScorer(
        model_id=args.aldi_model,
        device=args.device,
        local_only=args.local_only,
    )
    aldi_cascaded = scorer.score_texts(transcripts, batch_size=args.aldi_batch_size)

    pool_df = pd.DataFrame(
        {
            "sample_id": [s.sample_id for s in pool],
            "dialect": [s.dialect for s in pool],
            "n_tokens_gold": [len(s.reference_text.split()) for s in pool],
            "duration_sec": [round(s.duration_sec, 2) for s in pool],
            "gold_transcript": [s.reference_text for s in pool],
            "cascaded_transcript": transcripts,
            "wer": [round(word_error_rate(s.reference_text, t), 4) for s, t in zip(pool, transcripts)],
            "aldi_gold": [round(s.true_aldi, 4) for s in pool],
            "aldi_cascaded": [round(a, 4) for a in aldi_cascaded],
            "aldi_direct": [round(a, 4) for a in aldi_direct],
        }
    )
    pool_df["cascaded_minus_gold"] = (pool_df["aldi_cascaded"] - pool_df["aldi_gold"]).round(4)
    pool_df["direct_minus_gold"] = (pool_df["aldi_direct"] - pool_df["aldi_gold"]).round(4)
    pool_df = pool_df[COLUMNS]

    final_df = select_final(pool_df, args.samples_per_dialect)

    print()
    preview = [
        "sample_id", "dialect", "n_tokens_gold", "duration_sec",
        "wer", "aldi_gold", "aldi_cascaded", "aldi_direct",
    ]
    print(final_df[preview].to_string(index=False))
    print()
    write_outputs(final_df, pool_df, args.output_dir)


if __name__ == "__main__":
    main()
