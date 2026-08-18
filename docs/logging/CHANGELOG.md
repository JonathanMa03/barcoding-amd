# Detector development changelog

This file preserves detector experiments, parameter sweeps, and decisions that
are intentionally omitted from the streamlined README. Metrics are
column-level micro-aggregates over the ten scans in
`results/manual_ground_truth/`. Uncertain and vessel/structural annotations are
excluded from scoring.

These scans were used during parameter selection. The numbers describe the
development set and are not an estimate of performance on independent patients.

## 2026-08-18 — Selected configuration

The canonical configuration is `DETECTOR_CONFIG_0818` in
`src/detector/detector.py`.

- EA: contextual V3 classifier without an EA depth gate.
- Barcoding: contextual V3 plus Gabor mean/peak thresholds `0.40/0.50` and a
  near-depth mean z-score threshold of `0.0`.
- Structural model and dark-shadow veto enabled.
- Adjacent-B-scan evidence is explanatory metadata, not a mandatory veto.
- Hybrid rejection is not part of the selected detector.

Selected development metrics:

| Phenotype | Precision | Recall | Dice |
|---|---:|---:|---:|
| Barcoding, Gabor + depth | 0.817 | 0.520 | 0.636 |
| EA, contextual V3 | 0.475 | 0.699 | 0.566 |

The combined validation run assigns seven rejected barcoding columns to EA in
one case, producing EA precision `0.462` and Dice `0.557` when both selected
phenotype paths are run together. This class-interaction effect remains under
investigation.

## Single-scan detector evolution

| Phenotype | Experiment | Precision | Recall | Dice |
|---|---|---:|---:|---:|
| Barcoding | V2 calibrated | 0.526 | 0.506 | 0.516 |
| Barcoding | V3 contextual | 0.584 | 0.520 | 0.550 |
| Barcoding | Initial Gabor | 0.777 | 0.458 | 0.576 |
| Barcoding | Gabor + depth | **0.817** | **0.520** | **0.636** |
| EA | V2 calibrated | 0.369 | 0.739 | 0.492 |
| EA | V3 contextual | **0.475** | 0.699 | **0.566** |
| EA | Combined barcoding run | 0.462 | 0.699 | 0.557 |

### V2 calibrated classifiers

Separate logistic classifiers were calibrated for EA and barcoding. Barcoding
used probability `0.55`, minimum width 8, and maximum gap 4. EA used
probability `0.50`, minimum width 12, and maximum gap 5. V2 produced 371
false-positive barcoding columns and 222 false-positive EA columns.

### V3 contextual detector

V3 added probability smoothing, local supporting columns, feature voting,
whole-interval mean/peak probability, an independently calibrated structural
veto, and an explicit dark-shadow rule. False-positive columns fell to 302 for
barcoding and 136 for EA.

### Additional normal-rejection model

An additional learned normal/structural rejection model did not materially
improve quantitative or visual results and sometimes suppressed plausible
disease. It was rolled back. Its implementation is retained for provenance and
future experiments but is not selected.

## Gabor and depth parameter sweep

The first E2E Gabor gate required interval mean z-score `0.0` and peak z-score
`1.0`. It improved precision but reduced barcoding recall from `0.520` to
`0.458`.

A sweep of saved interval evidence selected:

- Gabor interval mean z-score: `0.40`
- Gabor interval peak z-score: `0.50`
- Near-depth interval mean z-score: `0.0`

Leave-one-patient-out threshold selection chose the same settings for every
held-out subject. This is encouraging stability within the small development
set, not external validation. The provisional EA depth gate did not improve
results and remains disabled.

The selected combined gate retained 424 true-positive barcoding columns—the
same as V3—while reducing false positives from 302 to 95.

## Adjacent-B-scan experiments

Two B-scans before and after each target were processed with the selected
single-scan detector.

| Rule | Barcoding precision | Recall | Dice | EA Dice |
|---|---:|---:|---:|---:|
| Single-scan control | 0.817 | 0.520 | **0.636** | **0.557** |
| Any one neighbor | 0.872 | 0.479 | 0.618 | 0.557 |
| Class-specific support | 0.872 | 0.479 | 0.618 | 0.518 |
| Bilateral support | 0.982 | 0.336 | 0.501 | 0.453 |
| Tighter spatial match | 0.872 | 0.479 | 0.618 | 0.557 |

One-neighbor support removed 38 false-positive and 34 true-positive barcoding
columns. Bilateral support was highly precise but removed too much annotated
disease. Tightening overlap from 25% to 50% and center shift from 30 to 20
columns made no difference. Adjacent support is retained as a confidence field
rather than a hard gate.

## Hybrid rejection experiments

The hybrid rules required no adjacent support, short width, and at least three
weak findings among mean Gabor energy, peak Gabor energy, near-depth brightness,
and mean lesion probability.

| Rule | TP | FP | FN | Precision | Recall | Dice |
|---|---:|---:|---:|---:|---:|---:|
| Single-scan control | 424 | 95 | 391 | 0.817 | 0.520 | **0.636** |
| Conservative, width ≤12 | 405 | 95 | 410 | 0.810 | 0.497 | 0.616 |
| Balanced, width ≤16 | 390 | 77 | 425 | 0.835 | 0.479 | 0.608 |

The conservative rule removed 19 true-positive columns and no false-positive
columns. The balanced rule removed 18 false-positive but 34 true-positive
columns. Some false intervals had stronger existing evidence than true
unsupported intervals, so recombining the same features did not solve
structural rejection.

## Current limitations and next experiment

- Ten non-clinician-annotated scans are insufficient for definitive tuning.
- Residual false positives are concentrated in a few scans, especially
  `fast_41_bscan_059`.
- Persistent anatomy can satisfy texture, depth, and adjacent-volume checks.
- Mandatory consistency gates remove locally weak true lesions.

The next proposed feature is a local vertical intensity-profile or shadow-
polarity comparison. Vessel/opaque shadowing should decrease deeper signal,
whereas pathological hypertransmission should increase it. This adds new
anatomical evidence rather than another combination of existing scores.
