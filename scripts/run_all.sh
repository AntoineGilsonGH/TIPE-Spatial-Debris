#!/usr/bin/env bash
# Regenerate every figure of the README (add --rebuild to re-simulate the ML dataset).
set -euo pipefail
cd "$(dirname "$0")"
python 01_tipe_2022.py
python 02_shells.py
python 03_sindy.py
python 04_surrogate.py "$@"
