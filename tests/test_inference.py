"""Tests for the inference pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from app.inference import (
    MODEL_PATH,
    dice_coefficient,
    get_model,
    iou_metric,
    load_slice_from_bytes,
    predict,
)


def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"


def test_model_param_count(model):
    assert model.count_params() == 1_968_691


def test_model_input_shape(model):
    assert model.input_shape == (None, 240, 240, 4)


def test_model_output_shape(model):
    assert model.output_shape == (None, 240, 240, 3)


def test_load_slice_from_bytes(sample_h5):
    image, mask = load_slice_from_bytes(sample_h5)
    assert image.shape == (240, 240, 4)
    assert mask is not None
    assert mask.shape == (240, 240, 3)


def test_load_slice_from_bytes_no_mask(sample_h5_no_mask):
    image, mask = load_slice_from_bytes(sample_h5_no_mask)
    assert image.shape == (240, 240, 4)
    assert mask is None


def test_load_slice_bad_shape():
    import io

    import h5py

    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("image", data=np.zeros((100, 100, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="Expected image shape"):
        load_slice_from_bytes(buf.getvalue())


def test_predict_shape(sample_h5):
    image, _ = load_slice_from_bytes(sample_h5)
    pred = predict(image)
    assert pred.shape == (240, 240, 3)
    assert np.all((pred >= 0) & (pred <= 1))


def test_dice_iou_metrics():
    y_true = np.ones((1, 10, 10, 3), dtype=np.float32)
    y_pred = np.ones((1, 10, 10, 3), dtype=np.float32)
    assert float(dice_coefficient(y_true, y_pred).numpy()) > 0.99
    assert float(iou_metric(y_true, y_pred).numpy()) > 0.99
