# Scenario-1-Earth-Observation

Brief repository for preparing, filtering, and training models on Earth-observation image patches (Delhi area example).

**Key goals:** prepare geospatial image patches, generate labels, and train a baseline model for land-cover / scene classification experiments.

**Contents:**
- **data/**: raw and derived input data used for preprocessing and patch extraction.
- **outputs/**: processed artifacts, datasets, patch arrays and CSV summaries created by preprocessing scripts.
- **scripts/**: training and utility scripts (e.g., `scripts/train_model.py`).
- `grid_and_filter.py`: grid generation and image filtering utilities.
- `prepare_labels.py`: label preparation and CSV/GeoJSON exports used for training.

## Repository structure

- `data/`
	- `rgb/`: source RGB images (if available).
	- `delhi_airshed.geojson`, `delhi_ncr_region.geojson`: example region masks used in filtering.
- `outputs/`
	- `datasets/labels.csv`: assembled labels for patches.
	- `datasets/class_counts.csv`: class distribution summary.
	- `datasets/filtered_images.geojson`: list of selected image extents after filtering.
	- `datasets/patches/`: saved patch arrays (NumPy `.npy`) used for training.
- `scripts/` and top-level scripts: preprocessing and training entrypoints.

## Quickstart

1. Create and activate a Python virtual environment (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Inspect or place source imagery into `data/rgb/` and ensure geojson masks are present in `data/`.

4. Run preprocessing (examples — check individual scripts for options and arguments):

```powershell
# create grid and filter images
python grid_and_filter.py

# prepare labels and dataset CSVs
python prepare_labels.py
```

5. Train a model (example):

```powershell
python scripts/train_model.py
```

Note: each script may accept flags and arguments; open the script files for usage details or run `-h`.

## Outputs

- `outputs/datasets/patches/`: per-patch NumPy arrays for model input.
- `outputs/datasets/labels.csv`: label table with patch coordinates, filenames and class ids.
- `outputs/figures/`: diagnostic plots produced during preprocessing or training.

## Development notes

- This repository is intended as a lightweight pipeline for experimentation. The scripts are minimal and meant to be adapted to new datasets or classification targets.
- If you plan to run large-scale training, consider adding data loaders, configuration via a CLI or YAML, and experiment tracking (e.g., MLflow, Weights & Biases).

## Contributing

If you'd like to contribute: open an issue describing proposed changes or submit a pull request. Keep changes focused and include tests where appropriate.

## Contact

For questions, contact the repository owner or open an issue in the repo.# Scenario-1-Earth-Observation
