#!/usr/bin/env python3
"""Convert the best_unet/ directory model into a single best_unet_final.keras file."""

from pathlib import Path

from app.inference import convert_directory_model_to_keras

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "best_unet"
DST = ROOT / "best_unet_final.keras"


def main() -> None:
    convert_directory_model_to_keras(SRC, DST)


if __name__ == "__main__":
    main()
