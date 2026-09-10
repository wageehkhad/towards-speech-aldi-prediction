"""
Repository-relative paths shared by the scripts.

Every default path in this repo resolves through here, so the scripts can be
run from a clone without editing absolute paths into them.
"""

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SUBMISSION_ROOT = SCRIPT_DIR.parent

ASSETS_DIR = SUBMISSION_ROOT / "assets"
RESULTS_DIR = SUBMISSION_ROOT / "results"
ANALYSIS_DIR = SUBMISSION_ROOT / "analysis"

MODELS_DIR = ASSETS_DIR / "models"
SADA_DIR = ASSETS_DIR / "sada"
CASABLANCA_DIR = ASSETS_DIR / "casablanca"
MEDIASPEECH_DIR = ASSETS_DIR / "mediaspeech"
HF_CACHE_DIR = ASSETS_DIR / "hf_cache"


def path_str(path: Path) -> str:
    return str(path)
