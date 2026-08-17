# measure.py
#
# Responsibilities:
# - Calculate clinically useful measurements from detected intervals.
# - Calculate total detected width.
# - Count detected intervals.
# - Calculate mean, median, minimum, and maximum interval width.
# - Calculate the affected horizontal fraction.
# - Summarize detector scores within detected intervals.
#
# Inputs:
# - Detection result
# - Image width
# - Optional physical pixel-spacing metadata
#
# Outputs:
# - Structured barcode measurements in pixels
# - Optional measurements in physical units when valid spacing
#   metadata are available
#
# This module should not alter the detector mask or intervals.
