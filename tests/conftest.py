"""Shared test fixtures."""

from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np
import pytest

from app.inference import get_model


def make_fake_h5(
    has_mask: bool = True,
    shape: tuple[int, int, int] = (240, 240, 4),
) -> bytes:
    """Create a minimal valid BraTS-style .h5 file in memory."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("image", data=np.random.rand(*shape).astype(np.float32))
        if has_mask:
            f.create_dataset(
                "mask",
                data=np.eye(shape[0], shape[1], dtype=np.float32)[..., np.newaxis]
                * np.array([[[1.0, 0.0, 0.0]]]),
            )
    return buf.getvalue()


@pytest.fixture(scope="session")
def model():
    """Load the model once per test session."""
    return get_model()


@pytest.fixture
def sample_h5() -> bytes:
    return make_fake_h5(has_mask=True)


@pytest.fixture
def sample_h5_no_mask() -> bytes:
    return make_fake_h5(has_mask=False)
