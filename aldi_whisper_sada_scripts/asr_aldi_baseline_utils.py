"""
Shared utilities for ASR -> Sentence-ALDi baseline experiments.
"""

import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

try:
    from jiwer import cer, wer
except Exception:  # pragma: no cover - optional dependency at runtime
    cer = None
    wer = None

# Known non-fatal warnings emitted by current transformers+whisper pipeline internals.
warnings.filterwarnings(
    "ignore",
    message=r"The input name `inputs` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"You have passed task=transcribe, but also have set `forced_decoder_ids`.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The attention mask is not set and cannot be inferred from input.*",
    category=UserWarning,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    try:
        return float(pearsonr(x, y)[0])
    except Exception:
        return float("nan")


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    try:
        return float(spearmanr(x, y)[0])
    except Exception:
        return float("nan")


def normalize_score_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ALDi" in out.columns:
        return out
    for col in ["aldi_score", "label", "true_aldi", "aldi"]:
        if col in out.columns:
            return out.rename(columns={col: "ALDi"})
    return out


def resolve_path(path: str, manifest_dir: Optional[str] = None) -> str:
    if not isinstance(path, str) or not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path

    candidates = [path, os.path.join(os.getcwd(), path)]
    if manifest_dir:
        candidates.append(os.path.join(manifest_dir, path))
    candidates.append(os.path.join(PROJECT_ROOT, path))

    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return path


@dataclass
class Sample:
    dataset: str
    sample_id: str
    true_aldi: float
    reference_text: str
    audio_input: object
    audio_path: str
    duration_sec: float
    dialect: str


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_manifest_samples(
    manifest_csv: str,
    dataset_name: str,
    max_samples: int = 0,
) -> List[Sample]:
    manifest_path = resolve_path(manifest_csv)
    manifest_dir = os.path.dirname(manifest_path)
    df = pd.read_csv(manifest_path)
    df = normalize_score_col(df)
    if "ALDi" not in df.columns:
        raise ValueError("Manifest missing ALDi score column: {}".format(manifest_path))

    id_col = _first_existing_col(df, ["id", "seg_id", "uid"])
    text_col = _first_existing_col(df, ["text", "transcription", "transcript", "sentence"])
    path_col = _first_existing_col(df, ["path", "audio_path", "audio"])
    dur_col = _first_existing_col(df, ["duration_sec", "duration"])
    dialect_col = _first_existing_col(df, ["dialect"])

    if path_col is None:
        raise ValueError("Manifest missing audio path column (path/audio_path): {}".format(manifest_path))

    if max_samples and max_samples > 0:
        df = df.head(max_samples)

    samples: List[Sample] = []
    for i, row in df.iterrows():
        raw_path = str(row[path_col]).strip()
        audio_path = resolve_path(raw_path, manifest_dir=manifest_dir)
        sample_id = str(row[id_col]) if id_col and pd.notna(row[id_col]) else "{}_{}".format(dataset_name, i)
        txt = ""
        if text_col and pd.notna(row[text_col]):
            txt = str(row[text_col]).strip()
        dur = float(row[dur_col]) if dur_col and pd.notna(row[dur_col]) else float("nan")
        dialect = str(row[dialect_col]) if dialect_col and pd.notna(row[dialect_col]) else "unknown"
        samples.append(
            Sample(
                dataset=dataset_name,
                sample_id=sample_id,
                true_aldi=float(row["ALDi"]),
                reference_text=txt,
                audio_input=audio_path,
                audio_path=audio_path,
                duration_sec=dur,
                dialect=dialect,
            )
        )
    return samples


def iter_casablanca_hf_samples(
    hf_dataset_dir: str,
    split: str,
    metadata_csv: Optional[str] = None,
    max_samples: int = 0,
) -> Iterator[Sample]:
    ds_any = load_from_disk(resolve_path(hf_dataset_dir))
    ds = ds_any[split] if isinstance(ds_any, dict) else ds_any

    meta_map: Dict[str, Dict] = {}
    if metadata_csv:
        meta = normalize_score_col(pd.read_csv(resolve_path(metadata_csv)))
        if "id" in meta.columns:
            meta_map = {str(r["id"]): r for r in meta.to_dict(orient="records")}

    n = len(ds) if not max_samples else min(len(ds), max_samples)
    for i in range(n):
        row = ds[i]
        sid = str(row.get("id", row.get("seg_id", i)))
        meta = meta_map.get(sid, {})
        audio = row["audio"]
        ref_txt = str(meta.get("text", row.get("transcription", "")) or "").strip()
        dialect = str(meta.get("dialect", "unknown"))
        dur = meta.get("duration", row.get("duration", np.nan))
        yield Sample(
            dataset="casablanca",
            sample_id=sid,
            true_aldi=float(meta.get("ALDi", row.get("label"))),
            reference_text=ref_txt,
            audio_input={
                "array": audio["array"],
                "sampling_rate": int(audio["sampling_rate"]),
            },
            audio_path=str(meta.get("audio_path", audio.get("path", ""))),
            duration_sec=float(dur) if pd.notna(dur) else float("nan"),
            dialect=dialect,
        )


class SentenceALDiScorer:
    def __init__(
        self,
        model_id: str = "AMR-KELEG/Sentence-ALDi",
        device: str = "cuda",
        local_only: bool = False,
        clamp_output: bool = False,
    ):
        kwargs = {"local_files_only": True} if local_only else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
        dev = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.model = self.model.to(dev).eval()
        self.device = dev
        self.clamp_output = clamp_output

    def score_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        max_length: int = 256,
    ) -> List[float]:
        scores: List[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [str(t) for t in texts[i : i + batch_size]]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.squeeze(-1).detach().float().cpu().numpy()
                # Sentence-ALDi is a regression head and can land slightly outside
                # [0, 1]. Clamping is opt-in so the default keeps the raw scores.
                if self.clamp_output:
                    logits = np.clip(logits, 0.0, 1.0)
                scores.extend([float(x) for x in logits.tolist()])
        return scores


class WhisperASRTranscriber:
    def __init__(
        self,
        asr_model: str,
        device: str = "cuda",
        chunk_length_s: float = 30.0,
        local_only: bool = False,
    ):
        if device.startswith("cuda") and torch.cuda.is_available():
            try:
                device_idx = int(device.split(":")[1]) if ":" in device else 0
            except Exception:
                device_idx = 0
        else:
            device_idx = -1

        model_kwargs: Dict[str, object] = {}
        if local_only:
            model_kwargs["local_files_only"] = True
        if torch.cuda.is_available() and device_idx >= 0:
            model_kwargs["torch_dtype"] = torch.float16

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            chunk_length_s=chunk_length_s,
            device=device_idx,
            model_kwargs=model_kwargs,
        )
        # Avoid task-vs-forced_decoder_ids warning: we explicitly control task/language at call time.
        try:
            self.pipe.model.generation_config.forced_decoder_ids = None
        except Exception:
            pass
        try:
            self.pipe.model.config.forced_decoder_ids = None
        except Exception:
            pass

    def transcribe_batch(self, audio_inputs: Sequence[object], batch_size: int = 8) -> List[Dict[str, str]]:
        out_rows: List[Dict[str, str]] = []
        try:
            outputs = self.pipe(
                list(audio_inputs),
                batch_size=batch_size,
                generate_kwargs={"language": "arabic", "task": "transcribe"},
            )
            if isinstance(outputs, dict):
                outputs = [outputs]
            for o in outputs:
                txt = str(o.get("text", "") if isinstance(o, dict) else o).strip()
                out_rows.append({"text": txt, "error": ""})
            return out_rows
        except Exception as e:
            # Fallback item-by-item so one bad sample doesn't kill the whole run.
            for item in audio_inputs:
                try:
                    o = self.pipe(
                        item,
                        generate_kwargs={"language": "arabic", "task": "transcribe"},
                    )
                    txt = str(o.get("text", "") if isinstance(o, dict) else o).strip()
                    out_rows.append({"text": txt, "error": ""})
                except Exception as sub_e:
                    out_rows.append({"text": "", "error": "{}".format(sub_e)})
            return out_rows


def run_asr_sentence_aldi(
    samples: Iterable[Sample],
    asr: WhisperASRTranscriber,
    scorer: SentenceALDiScorer,
    output_csv: str,
    asr_batch_size: int = 8,
    aldi_batch_size: int = 64,
    max_length: int = 256,
    progress_desc: str = "baseline",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    ensure_dir(os.path.dirname(output_csv))

    rows: List[Dict] = []
    batch: List[Sample] = []
    pbar = tqdm(desc=progress_desc, unit="sample")

    def flush(cur_batch: List[Sample]) -> None:
        if not cur_batch:
            return

        inputs = [s.audio_input for s in cur_batch]
        asr_out = asr.transcribe_batch(inputs, batch_size=asr_batch_size)
        transcripts = [r["text"] for r in asr_out]
        score_mask = [bool(t.strip()) for t in transcripts]
        texts_to_score = [t for t in transcripts if t.strip()]
        scored = scorer.score_texts(texts_to_score, batch_size=aldi_batch_size, max_length=max_length)
        scored_iter = iter(scored)

        for s, a in zip(cur_batch, asr_out):
            transcript = a["text"]
            pred = float(next(scored_iter)) if transcript.strip() else float("nan")
            rows.append(
                {
                    "dataset": s.dataset,
                    "sample_id": s.sample_id,
                    "audio_path": s.audio_path,
                    "true_aldi": s.true_aldi,
                    "predicted_aldi": pred,
                    "reference_text": s.reference_text,
                    "asr_transcript": transcript,
                    "duration_sec": s.duration_sec,
                    "dialect": s.dialect,
                    "asr_error": a["error"],
                    "asr_ok": int(a["error"] == ""),
                    "text_scored": int(transcript.strip() != ""),
                }
            )

    for sample in samples:
        batch.append(sample)
        if len(batch) >= asr_batch_size:
            flush(batch)
            pbar.update(len(batch))
            batch = []
    if batch:
        flush(batch)
        pbar.update(len(batch))
    pbar.close()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    metrics = compute_metrics(df)
    return df, metrics


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {
        "n_total": int(len(df)),
        "n_scored": int(df["predicted_aldi"].notna().sum()) if "predicted_aldi" in df.columns else 0,
        "n_asr_failed": int((df["asr_ok"] == 0).sum()) if "asr_ok" in df.columns else 0,
    }
    valid = df[df["predicted_aldi"].notna() & df["true_aldi"].notna()].copy()
    if len(valid) == 0:
        out.update(
            {
                "pearson": float("nan"),
                "spearman": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "wer": float("nan"),
                "cer": float("nan"),
            }
        )
        return out

    pred = valid["predicted_aldi"].to_numpy(dtype=np.float64)
    true = valid["true_aldi"].to_numpy(dtype=np.float64)
    err = pred - true
    out.update(
        {
            "pearson": safe_pearson(pred, true),
            "spearman": safe_spearman(pred, true),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
        }
    )

    refs = valid["reference_text"].fillna("").astype(str).tolist()
    hyps = valid["asr_transcript"].fillna("").astype(str).tolist()
    usable = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    if wer is not None and cer is not None and usable:
        ref_list = [u[0] for u in usable]
        hyp_list = [u[1] for u in usable]
        try:
            out["wer"] = float(wer(ref_list, hyp_list))
        except Exception:
            out["wer"] = float("nan")
        try:
            out["cer"] = float(cer(ref_list, hyp_list))
        except Exception:
            out["cer"] = float("nan")
    else:
        out["wer"] = float("nan")
        out["cer"] = float("nan")

    return out


def save_metrics_summary(path: str, metrics_by_dataset: Dict[str, Dict[str, float]]) -> None:
    ensure_dir(os.path.dirname(path))
    payload = {
        "datasets": metrics_by_dataset,
        "overall": {
            "n_total": int(sum(m.get("n_total", 0) for m in metrics_by_dataset.values())),
            "n_scored": int(sum(m.get("n_scored", 0) for m in metrics_by_dataset.values())),
            "n_asr_failed": int(sum(m.get("n_asr_failed", 0) for m in metrics_by_dataset.values())),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
