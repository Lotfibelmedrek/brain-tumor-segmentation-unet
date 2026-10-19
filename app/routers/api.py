"""FastAPI routes for the segmentation demo."""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import List

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from app.config import RESULTS_DIR, STATIC_DIR, TEMPLATES_DIR
from app.inference import load_slice_from_bytes, predict, score_prediction
from app.visualization import image_to_base64, make_overlay_image

router = APIRouter()

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _validate_h5(file: UploadFile) -> None:
    if not file.filename or not file.filename.lower().endswith(".h5"):
        raise HTTPException(status_code=400, detail="Only .h5 files are accepted.")


def _process_single(data: bytes, filename: str) -> dict:
    """Run inference on one uploaded .h5 file and return a result dict."""
    try:
        image, mask = load_slice_from_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid .h5 file: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {exc}")

    prediction = predict(image)
    pred_thresholded = (prediction > 0.5).astype(np.float32)

    scores = None
    if mask is not None:
        scores = score_prediction(mask, pred_thresholded)

    img = make_overlay_image(image, mask, prediction, width=1000)
    b64 = image_to_base64(img)

    # Save result image
    result_id = str(uuid.uuid4())
    out_path = RESULTS_DIR / f"{result_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    return {
        "id": result_id,
        "filename": filename,
        "has_ground_truth": mask is not None,
        "scores": scores,
        "image_url": f"/static/results/{result_id}.png",
        "preview": b64,
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the demo frontend."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend template not found.")
    return index_path.read_text(encoding="utf-8")


@router.post("/api/predict")
async def predict_single(file: UploadFile = File(...)) -> dict:
    """Run segmentation on a single .h5 slice."""
    _validate_h5(file)
    data = await file.read()
    return _process_single(data, file.filename or "unknown.h5")


@router.post("/api/predict/bulk")
async def predict_bulk(files: List[UploadFile] = File(...)) -> dict:
    """Run segmentation on multiple .h5 slices."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files per batch.")

    results = []
    errors = []
    for upload in files:
        try:
            _validate_h5(upload)
            data = await upload.read()
            results.append(_process_single(data, upload.filename or "unknown.h5"))
        except HTTPException as exc:
            errors.append({"filename": upload.filename, "error": exc.detail})
        except Exception as exc:
            errors.append({"filename": upload.filename, "error": str(exc)})

    return {"results": results, "errors": errors, "count": len(results)}


@router.get("/api/download/{result_id}")
async def download_result(result_id: str) -> FileResponse:
    """Download a single result PNG."""
    path = RESULTS_DIR / f"{result_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found.")
    return FileResponse(path, media_type="image/png", filename=f"{result_id}.png")


@router.post("/api/download/zip")
async def download_zip(payload: dict) -> StreamingResponse:
    """Download a ZIP archive of selected result PNGs."""
    ids = payload.get("ids", [])
    if not ids:
        raise HTTPException(status_code=400, detail="No result IDs provided.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for result_id in ids:
            path = RESULTS_DIR / f"{result_id}.png"
            if path.exists():
                zf.write(path, arcname=f"{result_id}.png")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=segmentation_results.zip"},
    )
