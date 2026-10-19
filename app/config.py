"""Application configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "best_unet_final.keras"
RESULTS_DIR = ROOT / "static" / "results"
STATIC_DIR = ROOT / "static"
TEMPLATES_DIR = ROOT / "app" / "templates"
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def get_temp_dir() -> Path:
    return Path(tempfile.gettempdir())
