#!/bin/bash

python3 -m venv .venv

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -m ipykernel install --user \
  --name barcoding-amd \
  --display-name "Python (barcoding-amd)"

# Windows users should follow the PowerShell commands in README.md.
