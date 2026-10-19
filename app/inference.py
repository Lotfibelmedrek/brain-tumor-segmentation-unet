"""Brain tumor segmentation inference utilities.

Loads the trained 2D U-Net and runs segmentation on BraTS 2020 axial slices.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import tensorflow as tf

# Reproducibility
SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------------------------------------------------------- #
# Metrics — must match the definitions used during training.
# --------------------------------------------------------------------------- #
def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.5) -> tf.Tensor:
    d_loss = dice_loss(y_true, y_pred)
    ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return alpha * d_loss + (1 - alpha) * tf.reduce_mean(ce_loss)


def iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(tf.round(y_pred), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


CUSTOM_OBJECTS = {
    "dice_coefficient": dice_coefficient,
    "dice_loss": dice_loss,
    "combined_loss": combined_loss,
    "iou_metric": iou_metric,
}

# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
MODEL_PATH = Path(__file__).resolve().parents[1] / "best_unet_final.keras"
FALLBACK_DIR = Path(__file__).resolve().parents[1] / "best_unet"


class ModelLoader:
    """Lazy-loading singleton for the Keras model."""

    _instance: "ModelLoader | None" = None
    _model: tf.keras.Model | None = None

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> tf.keras.Model:
        if self._model is None:
            model_path = self._resolve_model_path()
            self._model = tf.keras.models.load_model(
                str(model_path),
                custom_objects=CUSTOM_OBJECTS,
                compile=False,
            )
        return self._model

    @staticmethod
    def _resolve_model_path() -> Path:
        if MODEL_PATH.exists():
            return MODEL_PATH
        if FALLBACK_DIR.exists():
            return FALLBACK_DIR
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH} or {FALLBACK_DIR}. "
            "Place best_unet_final.keras in the project root."
        )


def get_model() -> tf.keras.Model:
    return ModelLoader().load()


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_slice_from_bytes(data: bytes) -> Tuple[np.ndarray, np.ndarray | None]:
    """Load a BraTS .h5 slice from bytes."""
    with h5py.File(io.BytesIO(data), "r") as f:
        if "image" not in f:
            raise ValueError("The .h5 file has no 'image' dataset.")
        image = f["image"][:].astype(np.float32)
        mask = f["mask"][:].astype(np.float32) if "mask" in f else None

    if image.shape != (240, 240, 4):
        raise ValueError(f"Expected image shape (240,240,4), got {image.shape}")
    return image, mask


def load_slice_from_path(h5_path: str | Path) -> Tuple[np.ndarray, np.ndarray | None]:
    """Load a BraTS .h5 slice from disk."""
    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    with h5py.File(path, "r") as f:
        if "image" not in f:
            raise ValueError("The .h5 file has no 'image' dataset.")
        image = f["image"][:].astype(np.float32)
        mask = f["mask"][:].astype(np.float32) if "mask" in f else None
    if image.shape != (240, 240, 4):
        raise ValueError(f"Expected image shape (240,240,4), got {image.shape}")
    return image, mask


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def predict(image: np.ndarray) -> np.ndarray:
    """Run model inference on a single slice.

    Args:
        image: array of shape (240, 240, 4).

    Returns:
        prediction array of shape (240, 240, 3).
    """
    model = get_model()
    batch = np.expand_dims(image, axis=0)  # (1, 240, 240, 4)
    return model.predict(batch, verbose=0)[0]  # (240, 240, 3)


def score_prediction(mask: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Compute Dice and IoU scores."""
    dice = float(dice_coefficient(mask, prediction).numpy())
    iou = float(iou_metric(mask, prediction).numpy())
    return {"dice": round(dice, 4), "iou": round(iou, 4)}


# --------------------------------------------------------------------------- #
# Conversion helper
# --------------------------------------------------------------------------- #
def convert_directory_model_to_keras(src_dir: str | Path, dst_path: str | Path) -> None:
    """Load a Keras directory model and save it as .keras."""
    src_dir = Path(src_dir)
    dst_path = Path(dst_path)
    print(f"Loading model from {src_dir} ...")
    model = tf.keras.models.load_model(
        str(src_dir), custom_objects=CUSTOM_OBJECTS, compile=False
    )
    print(f"Parameters: {model.count_params():,}")
    print(f"Saving to {dst_path} ...")
    model.save(str(dst_path), save_format="keras")
    print("Done.")
