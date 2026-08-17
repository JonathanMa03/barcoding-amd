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

```bash
barcoding-amd/
├── data/                         # Local datasets (ignored by Git)
├── docs/                         # Notes, reports, references
├── notebooks/
│   ├── 01_data.ipynb
│   ├── 02_cnn.ipynb
│   └── 03_gradcam.ipynb
├── results/
│   ├── figures/
│   └── models/
├── src/
│   ├── dataset.py
│   ├── gradcam.py
│   ├── models.py
│   └── train.py
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

## Workstation Setup

Clone the repository:

```bash
git clone https://github.com/JonathanMa03/barcoding-amd.git
cd barcoding-amd
```

Create environment:

```bash
bash scripts/setup_env.sh
```

Check setup:

```bash
python scripts/check_setup.py
```

Launch jupyter:

```bash
jupyter lab
```

## Classical detector pipeline

The OCT pipeline is organized into five packages:

- `src/loading/`: E2E volumes, B-scans/metadata, and JSON/PNG pairs.
- `src/preprocess/`: flattening, cropping, normalization, and denoising.
- `src/detector/`: hypertransmission, EA, and barcoding detector logic.
- `src/evaluation/`: annotation loading and tuning metrics.
- `src/visualization/`: static plots, interactive viewers, and Grad-CAM.

Each stage has an editable `CONFIG` dictionary and a persisted handoff file:

```bash
python scripts/load_data.py
python scripts/preprocess_data.py
python scripts/run_detector.py
python scripts/evaluate_detector.py
python scripts/visualize_detector.py
```

The EA/barcoding/normal rules are exploratory detector categories and are not
validated clinical diagnoses.

when running on windows, once output is created, commit after setting user.name and user.email. On mac, pull from main, should show a down arrow
