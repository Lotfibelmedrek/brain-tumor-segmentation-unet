"""Tests for FastAPI endpoints."""

from __future__ import annotations

import io
import zipfile

from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import make_fake_h5

client = TestClient(app)


def test_index_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "Brain Tumor Segmentation Demo" in response.text


def test_predict_single():
    data = make_fake_h5(has_mask=True)
    response = client.post(
        "/api/predict",
        files={"file": ("slice_1.h5", io.BytesIO(data), "application/x-hdf5")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["filename"] == "slice_1.h5"
    assert result["has_ground_truth"] is True
    assert "scores" in result
    assert "dice" in result["scores"]
    assert "iou" in result["scores"]
    assert result["preview"].startswith("data:image/png;base64,")


def test_predict_single_no_mask():
    data = make_fake_h5(has_mask=False)
    response = client.post(
        "/api/predict",
        files={"file": ("slice_2.h5", io.BytesIO(data), "application/x-hdf5")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["has_ground_truth"] is False
    assert result["scores"] is None


def test_predict_wrong_extension():
    response = client.post(
        "/api/predict",
        files={"file": ("slice.txt", io.BytesIO(b"not h5"), "text/plain")},
    )
    assert response.status_code == 400


def test_predict_bulk():
    files = [
        ("files", (f"slice_{i}.h5", io.BytesIO(make_fake_h5(has_mask=True)), "application/x-hdf5"))
        for i in range(3)
    ]
    response = client.post("/api/predict/bulk", files=files)
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 3
    assert len(payload["results"]) == 3
    assert len(payload["errors"]) == 0


def test_download_result():
    # First create a result
    data = make_fake_h5(has_mask=True)
    response = client.post(
        "/api/predict",
        files={"file": ("slice_dl.h5", io.BytesIO(data), "application/x-hdf5")},
    )
    result = response.json()
    result_id = result["id"]

    dl = client.get(f"/api/download/{result_id}")
    assert dl.status_code == 200
    assert dl.headers["content-type"] == "image/png"


def test_download_zip():
    files = [
        ("files", (f"slice_z{i}.h5", io.BytesIO(make_fake_h5(has_mask=True)), "application/x-hdf5"))
        for i in range(2)
    ]
    response = client.post("/api/predict/bulk", files=files)
    ids = [r["id"] for r in response.json()["results"]]

    zip_response = client.post("/api/download/zip", json={"ids": ids})
    assert zip_response.status_code == 200
    assert zip_response.headers["content-type"] == "application/zip"
    buf = io.BytesIO(zip_response.content)
    with zipfile.ZipFile(buf) as zf:
        assert len(zf.namelist()) == 2
