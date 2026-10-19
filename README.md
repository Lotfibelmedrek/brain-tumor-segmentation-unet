# Brain Tumor Segmentation Demo

A modern, simple FastAPI web interface to demo a 2D U-Net trained for brain tumor segmentation on BraTS 2020 axial MRI slices.

> Final-year project (State Engineer + Master's in Artificial Intelligence and Data Sciences), ESTIN Béjaïa, Algeria.

## What it does

- Upload a single `.h5` BraTS slice or many at once.
- The backend runs the U-Net and returns a side-by-side figure:
  **MRI (T1) | MRI (FLAIR) | Ground Truth | U-Net Prediction**.
- Tumor sub-regions are overlaid in color:
  - Yellow = whole tumor / edema
  - Green = tumor core
  - Red = enhancing tumor
- If the `.h5` file contains a `mask` dataset, Dice and IoU scores are computed.
- Bulk results can be downloaded individually or as a ZIP archive.

## Model

- File: `best_unet_final.keras` (~8 MB)
- Input: `(240, 240, 4)` — [T1, T1ce, T2, FLAIR]
- Output: `(240, 240, 3)` — tumor sub-region probabilities
- Parameters: **1,968,691**

### Results

| Metric | Score  |
|--------|--------|
| Dice   | 0.8176 |
| IoU    | 0.7187 |

Trained on BraTS 2020 using Kaggle free-tier GPUs (2× Tesla T4).

## Data

The model was trained on the **BraTS 2020** dataset (Multimodal Brain Tumor Segmentation Challenge).
The full dataset is **not included** in this repository because of its size and its terms of use.

- Official source: [BraTS 2020 challenge](https://www.med.upenn.edu/cbica/brats2020/data.html)
  (access requires registration, see the [registration page](https://www.med.upenn.edu/cbica/brats2020/registration.html))

### Sample slices

The `samples/` folder contains a few `.h5` slices so you can try the demo right away,
without downloading the full dataset.

### Using your own slices

1. Download the dataset from the official source above.
2. Pick any `.h5` slice file (one axial slice with 4 modalities: T1, T1ce, T2, FLAIR).
3. Upload it from the web interface at <http://localhost:8000>.
   You do not need to place it in a specific folder.

> BraTS data belongs to its authors and is subject to the challenge's terms of use.
> If you use it, please cite the BraTS publications listed on the official page.

## Quick start

Make sure [uv](https://docs.astral.sh/uv/) is installed.

```bash
# 1. Install dependencies and run the FastAPI server
PYTHONPATH=. uv run python main.py
```

Then open <http://localhost:8000> in your browser and upload a file from `samples/`.

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH = "."
uv run python main.py
```

## Run tests

```bash
PYTHONPATH=. uv run pytest
```

## Project structure

```
.
├── app/
│   ├── inference.py          # model loading, metrics, prediction
│   ├── visualization.py      # figure generation
│   ├── config.py             # paths & settings
│   ├── main.py               # FastAPI app
│   ├── routers/api.py        # HTTP routes
│   └── templates/index.html  # frontend UI
├── scripts/
│   └── convert_model.py      # best_unet/ -> best_unet_final.keras
├── tests/
│   ├── test_inference.py
│   └── test_api.py
├── samples/                  # example .h5 slices for the demo
├── best_unet_final.keras     # trained model (single file)
├── static/results/           # generated result PNGs (not versioned)
└── pyproject.toml
```

## Notes

- Only `.h5` files in the BraTS 2020 slice format are accepted.
- The model runs on CPU by default (no GPU required).
- This is a research/educational demo, **not a medical device**. Do not use it for clinical decisions.

## Author

Lotfi Belmedrek — ESTIN Béjaïa, Algeria