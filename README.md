# Barcoding in Age-Related Macular Degeneration (AMD)

This repository contains exploratory research on detecting and quantifying barcoding (hypertransmission patterns) in optical coherence tomography (OCT) images of patients with age-related macular degeneration (AMD).

The project combines classical image analysis techniques, convolutional neural networks (CNNs), and explainable AI methods to investigate whether barcoding can be measured reliably and ultimately used as a prognostic marker for disease progression.

Author: Jonathan Ma

---

## Aim

The primary objective of this project is to develop computational methods for identifying and quantifying barcoding patterns observed in OCT scans of AMD patients.

Current goals include:

* Detecting barcoding and hypertransmission patterns in OCT images.
* Developing quantitative measurements of barcode extent and morphology.
* Investigating whether barcode-related features are associated with disease progression.

---

## Data Sources

### Current Dataset

Retinal OCT Image Classification – 8 Classes

* 24,000 OCT images
* 8 retinal disease categories
* Used as a proof-of-concept dataset for CNN training and explainability

### Future Data

The primary analysis will utilize AMD OCT scans containing:

* Barcoding / hypertransmission-positive cases
* Barcoding / hypertransmission-negative cases
* Volume scans
* Longitudinal progression information (when available)

---

## Methodology

### Phase 1: Exploratory Barcoding Quantification

Classical image-analysis methods were used to construct an exploratory barcoding index:

* Transmission profile analysis
* Gabor filtering
* Structure tensor analysis
* Anisotropy measurements
* Composite barcoding index

These methods were evaluated using ROI perturbation and bootstrap sensitivity analyses.

### Phase 2: CNN Proof-of-Concept

A transfer-learning pipeline was developed using:

* ResNet50
* ImageNet pretrained weights
* Layer-wise fine tuning
* Grad-CAM explainability

Results:

* Frozen backbone accuracy: ~86%
* Fine-tuned accuracy: ~93%

Grad-CAM visualizations demonstrated localization of clinically meaningful retinal structures and pathology-associated regions.

### Phase 3: Planned Work

Future work includes:

* Hypertransmission detection
* Barcoding-positive vs. barcoding-negative classification
* Bounding-box localization of barcode regions
* Automated feature extraction
* Quantification of barcode width and area
* Progression-risk modeling

---

## Repository Structure

```text
barcoding-amd/
├── data/                              # Local data (ignored by Git)
│   └── heyex/meta/                    # Suggested location for E2E files
├── notebooks/                         # Exploration and validation notebooks
├── results/                           # Generated artifacts and figures
├── scripts/
│   ├── load_data.py                   # Load one E2E scan or JSON/PNG pair
│   ├── batch_preprocess_e2e.py        # Batch E2E loading + preprocessing
│   ├── preprocess_data.py             # Preprocess one loaded artifact
│   ├── run_detector.py                # Run detector on one processed scan
│   ├── validation_test.py              # Run all 10 manually selected scans
│   ├── extract_numerical_results.py    # Quantify EA/barcoding intervals
│   ├── evaluate_detector.py           # Compare output with annotations
│   └── visualize_detector.py          # Save detector overlay figure
├── src/
│   ├── loading/                       # E2E and JSON/PNG loading
│   ├── preprocess/                    # Flatten, crop, normalize, denoise
│   ├── detector/                      # Detector and feature logic
│   ├── evaluation/                    # Metrics and annotations
│   └── visualization/                 # Plotting, viewers, and Grad-CAM
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Current Status

### Completed:

* OCT dataset acquisition and inspection
* CNN training with transfer learning
* Model evaluation
* Grad-CAM explainability analysis
* Initial barcoding quantification experiments

### In Progress:

* Acquisition of barcoding-positive and barcoding-negative AMD volume scans
* Development of automated barcoding detection pipelines
* Hypertransmission localization and quantification

---

## 1. Environment setup from scratch

Python 3.10 or newer is recommended. Run all commands from the repository
root.

Clone the repository:

```bash
git clone https://github.com/JonathanMa03/barcoding-amd.git
cd barcoding-amd
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optionally register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name barcoding-amd --display-name "Python (barcoding-amd)"
```

Verify the installation:

```bash
python scripts/check_setup.py
```

After activation, commands can use `python`. Without activation, use
`.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows.

## 2. Loading data and choosing a processing scope

### Process one source

Edit `CONFIG` in `scripts/load_data.py`. For an E2E volume, select either its
central B-scan or an explicit index:

```python
CONFIG = {
    "source_path": Path("data/heyex/meta/example.E2E"),
    "metadata_path": None,
    "output_path": Path("results/pipeline/loaded_scan.npz"),
    "e2e_options": {
        "selection": "center",  # or "index"
        "bscan_index": None,     # set an integer when selection="index"
        "layer_name": "BM",
    },
    "source_metadata": {
        "progression_group": "fast",
        "subject_id": 8,
    },
}
```

For a PNG with a JSON metadata or annotation file:

```python
CONFIG = {
    "source_path": Path("data/example.png"),
    "metadata_path": Path("data/example.json"),
    "output_path": Path("results/pipeline/loaded_scan.npz"),
    "e2e_options": {},
}
```

Run:

```bash
python scripts/load_data.py
```

The output is a compressed `.npz` artifact containing the image, source
metadata, B-scan index, and retinal-layer boundary when available.

### Batch-load and preprocess E2E data

`scripts/batch_preprocess_e2e.py` loads and preprocesses without creating a
separate loaded artifact for each scan. Put E2E files in `data/heyex/meta/`,
then choose one `BATCH_CONFIG["mode"]`:

- `"volume_all_scans"`: every B-scan from the E2E file in `volume_path`.
- `"all_volumes_selected_scan"`: one center or indexed B-scan from every E2E
  file in `input_directory`.
- `"all_volumes_all_scans"`: every B-scan from every E2E file.

The relevant batch options are:

- `volume_path`: the single E2E file used by `volume_all_scans`.
- `input_directory`: directory searched by the two all-volume modes.
- `recursive`: search nested directories when `True`.
- `scan_selection`: `"center"` or `"index"` for the selected-scan mode.
- `selected_bscan_index`: index used when `scan_selection="index"`.
- `output_directory`: root directory for processed `.npz` artifacts.
- `overwrite`: replace existing scan artifacts when `True`; otherwise skip.
- `continue_on_error`: record failed scans and continue when `True`.

Run:

```bash
python scripts/batch_preprocess_e2e.py
```

Outputs are grouped by volume:

```text
results/batch_preprocessed/
├── volume_name/
│   ├── bscan_0000.npz
│   ├── bscan_0001.npz
│   └── ...
└── manifest.json
```

The manifest records the source file, B-scan index, output path, output shape,
and whether each scan was processed, skipped, or failed.

## 3. Preprocessing configuration

For a single loaded artifact, edit `PIPELINE_CONFIG` and
`PREPROCESSING_CONFIG` in `scripts/preprocess_data.py`, then run:

```bash
python scripts/preprocess_data.py
```

For batch E2E preprocessing, edit the identically named
`PREPROCESSING_CONFIG` in `scripts/batch_preprocess_e2e.py`. These settings
mirror `pipeline_validation.ipynb`:

- `layer_name`: annotated retinal boundary used for alignment, normally
  `"BM"` for Bruch's membrane. The selected layer must exist in the E2E file.
- `reference_row`: row to which the boundary is flattened. `None` uses the
  median boundary location. A larger number positions the flattened boundary
  lower in the image.
- `flatten_fill_value`: value inserted into empty pixels created by vertically
  shifting image columns.
- `depth_below_layer`: number of rows retained at and below the flattened
  boundary. Larger values include more deep tissue and choroid.
- `include_boundary`: include the boundary row when `True`; start one row below
  it when `False`.
- `require_full_depth`: raise an error if the requested crop is unavailable
  when `True`; pad a shallow crop when `False`.
- `crop_fill_value`: value used for padded crop rows.
- `normalization_method`: `"zscore"`, `"percentile"`, `"minmax"`, or
  `"none"`. Z-score normalization centers and scales by standard deviation;
  percentile and min-max normalization produce values in `[0, 1]`.
- `lower_percentile`: lower clipping percentile used only by percentile
  normalization. Increasing it ignores more dark outliers.
- `upper_percentile`: upper clipping percentile used only by percentile
  normalization. Lowering it clips more bright outliers.
- `denoise_method`: `"gaussian"`, `"median"`, or `"none"`.
- `gaussian_sigma`: Gaussian smoothing strength as either one value or
  `(depth_sigma, horizontal_sigma)`. Larger values remove more noise but blur
  finer structures.

The preprocessing API also supports `zscore_epsilon`, `median_size`, and
`gaussian_mode` when those methods need further control. The resulting artifact
defaults to `results/pipeline/preprocessed_scan.npz`.

## 4. Running and configuring the detector

Edit `PIPELINE_CONFIG` and `DETECTOR_CONFIG` in `scripts/run_detector.py`:

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

Structural detector parameters:

- `detector_type`: `"structural"` runs the structural-hypertransmission
  pipeline. `"weighted"` selects the alternative weighted-feature detector.
- `verticality_smoothing_sigma`: smoothing before image-gradient calculation.
  Larger values reduce speckle but can blur fine vertical structures.
- `verticality_threshold`: minimum vertical organization required for a pixel
  to enter the structural mask. Larger values are more selective.
- `gradient_quantile`: gradient-magnitude quantile used to retain structural
  pixels. Larger values retain only stronger edges.
- `minimum_component_size`: minimum structural connected-component area in
  pixels; `0` disables component-size filtering.
- `column_upper_quantile`: upper intensity statistic calculated from each
  structurally cleaned column. Larger values emphasize its brightest pixels.
- `minimum_valid_pixels`: minimum number of non-structural pixels required for
  a column-level intensity estimate.
- `signal_smoothing_sigma`: Gaussian smoothing applied to one-dimensional
  column signals. Larger values suppress short local changes.
- `median_iqr_multiplier`: number of reference IQRs added to the median-column
  baseline. Larger values require stronger hypertransmission.
- `q90_iqr_multiplier`: equivalent IQR multiplier for the upper-quantile
  intensity signal.
- `continuity_window_width`: horizontal width used to evaluate local depth
  continuity.
- `continuity_depth_lag`: maximum vertical displacement considered when
  matching neighboring columns.
- `continuity_minimum_row_standard_deviation`: numerical floor that prevents
  unstable correlation in nearly constant image regions.
- `continuity_quantile`: scan-relative continuity threshold. Larger values
  retain fewer, more continuous candidates.
- `vertical_fraction_quantile`: scan-relative threshold for the fraction of
  vertically organized pixels in candidate columns.
- `minimum_positive_run`: minimum retained horizontal detection length in
  columns.
- `maximum_negative_gap`: largest internal negative gap filled between nearby
  detections; `0` disables gap filling.
- `edge_margin`: number of columns excluded at both lateral image edges.

EA and barcoding classification is performed by two independent calibrated
models in `phenotype_config`. Each class has:

- `probability_threshold`: minimum class probability; increasing it improves
  selectivity while generally reducing recall.
- `minimum_positive_run`: minimum retained interval width for that class.
- `maximum_negative_gap`: largest internal gap joined for that class.
- `coefficients` and `intercept`: fitted calibration values derived from the
  manual annotations. These should normally remain unchanged unless the model
  is recalibrated.

`structural_veto` is a third calibrated model trained on the manual
`Vessel / Structural` annotations. It removes likely vessel-shadow columns
after EA/barcoding gap filling. Its probability threshold and cleanup settings
can be adjusted independently; higher thresholds make the veto more
conservative. `margin_columns` expands each excluded vessel run laterally to
cover the surrounding shadow; its default is two columns.

When both models select the same column, the detector compares each model's
probability relative to its threshold and assigns the stronger class.

Calibration v2 uses a barcoding probability threshold of `0.55` with an
8-column minimum interval and an EA threshold of `0.50` with a 12-column
minimum interval. These settings were selected to reduce the false-positive
fragmentation observed across the ten validation scans.

Calibration v3 adds spatial and anatomical rejection after the calibrated
column probabilities are computed:

- `smoothing_sigma` and `raw_probability_weight` blend each column's score
  with evidence from neighboring columns.
- `support_radius`, `support_probability_margin`, and
  `minimum_local_support` require a candidate to be surrounded by sustained
  near-threshold evidence instead of being an isolated intensity response.
- `feature_vote_thresholds` and `minimum_feature_votes` require agreement
  among scan-relative median intensity, continuity, and verticality. EA uses
  the stricter three-feature rule; barcoding retains a two-feature rule to
  avoid the large recall loss seen with three-way agreement.
- `minimum_interval_mean_probability` and
  `minimum_interval_peak_probability` reject weak candidates at the whole
  lesion level after cleanup and structural exclusion.
- `structural_veto.dark_shadow` rejects persistent dark, continuous columns
  that behave like vessel shadows rather than hypertransmission. Its
  `margin_columns` setting excludes the immediate shadow boundary as well.

The detector JSON metadata reports how many columns were rejected by local
support and feature voting, how many complete intervals failed lesion-level
confidence, and how many columns were covered by the dark-shadow veto.

Calibration v4 adds an explicit `normal_rejection` model after the independent
EA and barcoding decisions. It estimates the probability that each column is
normal from the same eleven scan-relative intensity and structural features
used by the anatomical rejection stage. A provisional probability threshold
of `0.50` is used; normal evidence must persist for at least 5 columns, and
gaps up to 2 columns may be joined. The provisional coefficients were fitted
on clean preprocessed PNG development scans. Results produced directly from
E2E data should be used to decide whether these coefficients and the `0.85`
threshold transfer adequately. Set `phenotype_config["normal_rejection"]` to
`None` to disable this stage for an A/B comparison.

The JSON output contains one `normal`, `ea`, or `barcoding` label per column,
detected intervals, thresholds, label counts, and the detector configuration.
EA and barcoding are exploratory research labels, not validated clinical
diagnoses.

### Extracting numerical results

After running the detector, extract the EA and barcoding interval counts and
widths with:

```bash
python scripts/extract_numerical_results.py
```

Edit `CONFIG` in that script if the detector output is stored elsewhere. By
default it reads `results/pipeline/detections.json` and writes a metadata-based
name such as `results/pipeline/fast_08_bscan_048_automatic.json`. The output contains the number of
intervals, each interval's inclusive start/end columns and width in pixels,
total width, and mean, median, minimum, and maximum width for both EA and
barcoding.

To run the detector on batch-preprocessed scans, set `input_path` and
`output_path` to one artifact at a time, for example:

```python
"input_path": Path("results/batch_preprocessed/volume_name/bscan_0048.npz"),
"output_path": Path("results/batch_detections/volume_name/bscan_0048.json"),
```

## 5. Visualization

Edit `CONFIG` in `scripts/visualize_detector.py` so the preprocessing and
detection paths describe the same B-scan:

```python
CONFIG = {
    "preprocessed_path": Path("results/pipeline/preprocessed_scan.npz"),
    "detection_path": Path("results/pipeline/detections.json"),
    "output_directory": Path("results/pipeline"),
    "identity_overrides": {
        "progression_group": None,
        "subject_id": None,
        "bscan_index": None,
    },
    "title": "EA and barcoding detector output",
    "colors": {"ea": "tab:orange", "barcoding": "tab:red"},
    "figure_options": {"figsize": (12, 4), "dpi": 150},
}
```

Run:

```bash
python scripts/visualize_detector.py
```

The script saves a metadata-based filename such as
`fast_08_bscan_048_automatic.png`, matching its numerical-results JSON. It
shows a grayscale preprocessed B-scan with transparent EA and
barcoding interval overlays. `colors`, `figsize`, `dpi`, title, and output path
can be changed without modifying plotting code.

The B-scan number is read from the processed artifact, so each patient can use
a different selected scan. When loading E2E data, set `source_metadata` in
`scripts/load_data.py` with `progression_group` and `subject_id`. Subject ID is
also inferred from E2E names such as `ea8.E2E`. If older artifacts lack this
metadata, set `identity_overrides` in the numerical and visualization scripts.

## Optional evaluation

To compare a detection with manual JSON annotations, edit `CONFIG` in
`scripts/evaluate_detector.py` and run:

```bash
python scripts/evaluate_detector.py
```

Available metrics include confusion counts, precision, sensitivity/recall,
specificity, accuracy, F1/Dice, intersection over union, and detected/target
fractions. Barcoding and EA are scored separately. Columns annotated as
`Uncertain` or `Vessel / Structural` are excluded from scoring rather than
treated as negative examples.

### Ten-scan validation workflow

To run the complete workflow on the ten patient-specific scans listed in
`results/manual_ground_truth/`, place the corresponding E2E files under
`data/heyex/meta/` with names such as `ea8.E2E`, then run:

```bash
python scripts/validation_test.py
```

The script reads each subject ID, progression group, and distinct B-scan index
from the manual JSON—not from a shared hardcoded scan number. It performs E2E
loading, preprocessing, calibrated detection, numerical quantification,
ground-truth evaluation, and plotting. Paired files are saved under
`results/automatic_detector/`, for example:

```text
fast_08_bscan_048_automatic.json
fast_08_bscan_048_automatic.png
```

`validation_manifest.json` records processed, skipped, and failed cases.
`overwrite` and `continue_on_error` can be changed in `VALIDATION_CONFIG`.

## Package responsibilities

The OCT pipeline is organized into five packages:

- `src/loading/`: E2E volumes, B-scans/metadata, and JSON/PNG pairs.
- `src/preprocess/`: flattening, cropping, normalization, and denoising.
- `src/detector/`: hypertransmission, EA, and barcoding detector logic and
  parameters.
- `src/evaluation/`: annotation loading and tuning metrics.
- `src/visualization/`: static plots, interactive viewers, and Grad-CAM.

Notebooks may use older imports while the refactoring is in progress. The
terminal scripts above are the canonical workflow.
