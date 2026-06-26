#!/bin/bash

python -m venv .venv
source .venv/Scripts/activate

#Windows use Scripts instead of bin

pip install --upgrade pip
pip install -r requirements.txt

python -m ipykernel install --user \
  --name barcoding-amd \
  --display-name "Python (barcoding-amd)"

#chmod +x scripts/setup_env.sh
# Window button + L for logout