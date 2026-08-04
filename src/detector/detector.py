# detector.py
#
# Responsibilities:
# - Validate and apply feature weights.
# - Calculate the combined barcode score.
# - Threshold the continuous score.
# - Remove short positive runs.
# - Fill short gaps between detections.
# - Extract contiguous horizontal detection intervals.
# - Return a structured detection result.
#
# Future responsibilities:
# - Tune feature weights against manual annotations.
# - Learn weights using logistic regression.
# - Select detection thresholds using validation data.
#
# Inputs:
# - Feature signals
# - Feature weights
# - Threshold and spatial-cleanup parameters
#
# Outputs:
# - Combined score
# - Raw and cleaned masks
# - Contiguous detection intervals
# - Detector metadata
#
# This module should not load data or produce figures.