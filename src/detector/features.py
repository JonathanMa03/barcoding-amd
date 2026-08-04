# features.py
#
# Responsibilities:
# - Accept a preprocessed two-dimensional image.
# - Construct overlapping horizontal sliding windows.
# - Support configurable window width, stride, and padding.
# - Calculate interpretable mathematical feature signals,
#   including verticality, depth persistence, periodicity,
#   amplitude, and spatial heterogeneity.
# - Standardize feature signals before combination.
#
# Inputs:
# - Preprocessed image
# - Sliding-window configuration
# - Feature-specific parameters
#
# Outputs:
# - One feature value per horizontal location for each feature
# - Raw and standardized feature signals
# - Feature-extraction metadata
#
# This module should describe the image mathematically but should
# not apply feature weights, thresholds, or class labels.