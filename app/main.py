"""FastAPI entry point for the brain tumor segmentation demo."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import MODEL_PATH, STATIC_DIR
from app.inference import get_model
from app.routers import api


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the model on startup to fail fast if it is missing."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run: PYTHONPATH=. uv run python scripts/convert_model.py"
        )
    print(f"Loading model from {MODEL_PATH} ...")
    model = get_model()
    print(f"Model loaded: {model.count_params():,} parameters")
    yield


app = FastAPI(
    title="Brain Tumor Segmentation Demo",
    description="U-Net segmentation demo for BraTS 2020 axial slices.",
    version="1.0.0",
    lifespan=lifespan,
)

# Static files (result images)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# API + frontend routes
app.include_router(api.router)
