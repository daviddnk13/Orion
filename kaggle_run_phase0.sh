#!/bin/bash
# Kaggle one-liner for Phase 0 SOL mapping test
set -e
pip install lightgbm scipy requests matplotlib -q
git clone https://github.com/daviddnk13/Orion.git /tmp/orion
cd /tmp/orion && python3 sol_phase0_mappings.py
