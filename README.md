# EA and barcoding detection in AMD OCT

This repository contains an interpretable research pipeline for locating and
quantifying Early Atrophy (EA) and barcoding patterns in retinal OCT B-scans.
It accepts native HEYEX E2E volumes or PNG/JSON pairs and produces:

- one `normal`, `ea`, or `barcoding` label per image column;
- EA and barcoding interval counts, locations, and widths;
- PNG overlays of the detected intervals;
- JSON evidence describing feature values and rejection decisions.

The canonical detector is `DETECTOR_CONFIG_0818` in
`src/detector/detector.py`. It uses contextual V3 for EA and contextual V3 plus
selected Gabor-texture and near-depth evidence for barcoding. This is
exploratory research software, not a clinically validated diagnostic system.

Historical experiments and parameter sweeps are recorded in
[`docs/logging/CHANGELOG.md`](docs/logging/CHANGELOG.md).

## Repository layout

```text
barcoding-amd/
├── data/heyex/meta/              # Suggested local E2E location
├── scripts/                      # Terminal workflow and experiments
├── src/
│   ├── loading/                  # E2E and PNG/JSON loading
│   ├── preprocess/               # Flatten, crop, normalize, denoise
│   ├── detector/                 # Features, selected detector, experiments
│   ├── evaluation/               # Annotation and metric utilities
│   └── visualization/            # Detection plots and viewers
├── results/                      # Generated outputs and validation data
├── docs/logging/                 # Experiment history
├── docs/notes/                   # Detailed technical notes
└── docs/presentations/           # Project slide decks
```

## 1. Environment setup

Python 3.10 or newer is recommended. Run commands from the repository root.

```bash
git clone https://github.com/JonathanMa03/barcoding-amd.git
cd barcoding-amd
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_setup.py
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## 2. Canonical single-scan workflow

The standard terminal workflow is:

```text
load_data.py → preprocess_data.py → run_detector.py
             → extract_numerical_results.py → visualize_detector.py
```

Each script contains a small editable path/configuration dictionary near the
top. The detector itself should normally remain `DETECTOR_CONFIG_0818`.

### Step A: Load an E2E B-scan

Place E2E volumes under `data/heyex/meta/`. Edit `CONFIG` in
`scripts/load_data.py`:

```python
CONFIG = {
    "source_path": Path("data/heyex/meta/ea8.E2E"),
    "metadata_path": None,
    "output_path": Path("results/pipeline/loaded_scan.npz"),
    "e2e_options": {
        "selection": "index",
        "bscan_index": 48,
        "layer_name": "BM",
    },
    "source_metadata": {
        "progression_group": "fast",
        "subject_id": 8,
    },
}
```

Then run:

```bash
python scripts/load_data.py
```

To load the middle scan automatically, use `"selection": "center"` and set
`bscan_index` to `None`.

### PNG + JSON input

The same loader accepts an image with its metadata or annotation JSON:

```python
CONFIG = {
    "source_path": Path("data/example.png"),
    "metadata_path": Path("data/example.json"),
    "output_path": Path("results/pipeline/loaded_scan.npz"),
    "e2e_options": {},
}
```

PNG images without a retinal boundary are treated as already selected or
cropped images. Normalization and denoising still run.

### Step B: Preprocess

Edit the paths in `scripts/preprocess_data.py`, then run:

```bash
python scripts/preprocess_data.py
```

The selected preprocessing settings are:

| Parameter | Selected value | Meaning |
|---|---:|---|
| `layer_name` | `"BM"` | Anatomical boundary used for alignment |
| `reference_row` | `None` | Flatten to the scan's median BM row |
| `depth_below_layer` | `150` | Retain 150 rows at and below BM |
| `include_boundary` | `True` | Include the BM row in the crop |
| `require_full_depth` | `False` | Pad a shallow crop instead of failing |
| `normalization_method` | `"zscore"` | Express brightness relative to the scan |
| `denoise_method` | `"gaussian"` | Apply mild Gaussian denoising |
| `gaussian_sigma` | `(1.0, 0.5)` | Smooth more through depth than horizontally |

Other supported normalization methods are `percentile`, `minmax`, and `none`.
Other denoising choices are `median` and `none`. These alternatives are not
part of the selected workflow.

### Optional: create clinician manual ground truth

Run both loading and preprocessing before opening the annotator:

```bash
python scripts/load_data.py
python scripts/preprocess_data.py
python scripts/manual_annotator.py ea8 48 --group fast
```

The patient can be written as `ea8` or `8`; the scan number is the zero-based
B-scan index. `--group` may be omitted when `progression_group` was included in
`scripts/load_data.py`. The annotator checks that the requested patient and
scan agree with the preprocessed artifact.

In the annotation window:

1. Select `Early Atrophy (EA)`, `Barcoding`, or `Normal`.
2. Drag horizontally across the B-scan to mark an interval.
3. Use `Undo last` or `Clear all` to correct annotations.
4. Press `Save PNG + JSON` when finished.

The paired files are saved under `results/manual_ground_truth/`:

```text
fast_08_bscan_048_ground_truth.json
fast_08_bscan_048_ground_truth.png
```

Existing ground truth is protected by default. Pass `--overwrite` only when
you intend to replace both files. To annotate an artifact stored elsewhere,
use `--input path/to/preprocessed_scan.npz`.

### Step C: Run `DETECTOR_CONFIG_0818`

Edit only the input and output paths in `scripts/run_detector.py`:

```python
PIPELINE_CONFIG = {
    "input_path": Path("results/pipeline/preprocessed_scan.npz"),
    "output_path": Path("results/pipeline/detections.json"),
}
```

Run:

```bash
python scripts/run_detector.py
```

The script imports a deep copy of the selected configuration:

```python
from src.detector.detector import DETECTOR_CONFIG_0818

DETECTOR_CONFIG = deepcopy(DETECTOR_CONFIG_0818)
```

The selected detector performs the following operations:

1. Measures scan-relative brightness, depth continuity, and vertical structure.
2. Scores EA and barcoding independently with contextual V3 classifiers.
3. Requires local support and whole-interval probability evidence.
4. Removes calibrated vessel/structural and dark-shadow candidates.
5. Requires Gabor stripe evidence and near-depth brightness for barcoding.
6. Reports final normal, EA, and barcoding labels with interval evidence.

Selected phenotype parameters:

| Setting | Barcoding | EA |
|---|---:|---:|
| Probability threshold | `0.55` | `0.50` |
| Minimum interval width | `8` | `12` |
| Maximum joined gap | `4` | `5` |
| Local support radius | `5` | `7` |
| Required feature votes | `2 of 3` | `3 of 3` |
| Gabor mean / peak | `0.40 / 0.50` | disabled |
| Depth gate | near mean ≥ `0.0` | disabled |

### Step D: Extract interval counts and widths

Edit `CONFIG` in `scripts/extract_numerical_results.py`, then run:

```bash
python scripts/extract_numerical_results.py
```

The output reports the number of EA and barcoding intervals, their inclusive
start/end columns, individual widths, total width, and width summary statistics.

### Step E: Create the overlay

Point `scripts/visualize_detector.py` to the preprocessed NPZ and detection JSON:

```bash
python scripts/visualize_detector.py
```

The output PNG uses orange for EA and red for barcoding. Metadata-based names
follow the format:

```text
fast_08_bscan_048_automatic.png
fast_08_bscan_048_automatic.json
```

## 3. Ten-scan selected validation workflow

To reproduce the current development validation:

1. Put the corresponding E2E files under `data/heyex/meta/`, named like
   `ea8.E2E`.
2. Keep the ten manual JSON files in `results/manual_ground_truth/`.
3. Run:

```bash
python scripts/validation_test.py
```

The script discovers each subject and patient-specific B-scan index from the
manual JSON, then performs loading, preprocessing, `DETECTOR_CONFIG_0818`,
quantification, evaluation, and visualization. Results are written to
`results/automatic_detector_gabor_depth/` with a `validation_manifest.json`.

Columns annotated as `Uncertain` or `Vessel / Structural` are excluded from
scoring. The current development results are:

| Phenotype | Precision | Recall | Dice |
|---|---:|---:|---:|
| Barcoding | `0.817` | `0.520` | `0.636` |
| EA in combined run | `0.462` | `0.699` | `0.557` |

These ten scans informed parameter selection and do not constitute independent
clinical validation.

## 4. Batch preprocessing E2E volumes

`scripts/batch_preprocess_e2e.py` supports three modes:

- `volume_all_scans`: every scan in one E2E volume;
- `all_volumes_selected_scan`: one center or indexed scan per E2E file;
- `all_volumes_all_scans`: every scan from every E2E file.

Edit `BATCH_CONFIG` and run:

```bash
python scripts/batch_preprocess_e2e.py
```

Artifacts are grouped by volume under `results/batch_preprocessed/`, and
`manifest.json` records processed, skipped, and failed scans.

## 5. Evaluation utilities

To evaluate an individual detection JSON against a manual annotation, edit
`CONFIG` in `scripts/evaluate_detector.py` and run:

```bash
python scripts/evaluate_detector.py
```

Metrics include confusion counts, precision, recall, specificity, accuracy,
Dice, intersection over union, and predicted/target fractions.

## Experimental and historical implementations

Experimental code is intentionally retained in place for reproducibility. It
is marked in source comments and is not part of `DETECTOR_CONFIG_0818`.

- `scripts/adjacent_bscan_consistency.py`: mandatory adjacency, bilateral,
  spatially strict, and hybrid rejection experiments.
- `CALIBRATED_PHENOTYPE_V1_CONFIG`: contextual V3 calibration before selected
  Gabor/depth gates are enabled.
- `STRUCTURAL_HYPERTD_V1_CONFIG`: historical structural feature defaults.
- Weighted detector, normal-model, training, and Grad-CAM modules: retained for
  earlier experiments and future comparison.

Do not treat experimental scripts as the main inference workflow. See
[`docs/logging/CHANGELOG.md`](docs/logging/CHANGELOG.md) for parameter sweeps,
metrics, rejected approaches, and the rationale for the selected configuration.
