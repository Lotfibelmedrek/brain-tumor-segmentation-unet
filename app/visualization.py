"""Visualization helpers for brain tumor segmentation results."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

# BraTS convention: label 0 -> yellow, 1 -> green, 2 -> red
OVERLAY = ListedColormap([(1, 1, 0), (0, 1, 0), (1, 0, 0)])


def make_overlay_image(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray | None = None,
    width: int = 800,
) -> Image.Image:
    """Create the 4-column reference figure: T1 | FLAIR | GT | Prediction.

    Args:
        image: (240, 240, 4) MRI modalities.
        mask: (240, 240, 3) ground-truth mask or None.
        prediction: (240, 240, 3) predicted mask or None.
        width: output image width in pixels.

    Returns:
        PIL RGB image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_mask = mask is not None
    has_pred = prediction is not None
    n_cols = 2 + int(has_mask) + int(has_pred)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    t1 = image[:, :, 0]
    flair = image[:, :, 3]

    axes[0].imshow(t1, cmap="gray")
    axes[0].set_title("MRI (T1)", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(flair, cmap="gray")
    axes[1].set_title("MRI (FLAIR)", fontweight="bold")
    axes[1].axis("off")

    col = 2
    if has_mask:
        _overlay_on_axis(axes[col], flair, mask, "Ground Truth")
        col += 1
    if has_pred:
        _overlay_on_axis(axes[col], flair, prediction, "U-Net Prediction")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=width // (5 * n_cols), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _overlay_on_axis(
    ax, mri_gray: np.ndarray, mask3: np.ndarray, title: str
) -> None:
    """Plot a 3-channel mask overlay on a grayscale MRI axis."""
    ax.imshow(mri_gray, cmap="gray")
    labels = np.argmax(mask3, axis=-1).astype(float)
    present = mask3.sum(axis=-1) > 0.5
    masked = np.ma.masked_where(~present, labels)
    ax.imshow(masked, cmap=OVERLAY, alpha=0.55, vmin=0, vmax=2)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


def image_to_base64(img: Image.Image) -> str:
    """Encode a PIL image to a base64 data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def save_result_image(
    image: np.ndarray,
    mask: np.ndarray | None,
    prediction: np.ndarray | None,
    out_path: Path,
    width: int = 800,
) -> Path:
    """Save the overlay figure to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = make_overlay_image(image, mask, prediction, width=width)
    img.save(out_path)
    return out_path
