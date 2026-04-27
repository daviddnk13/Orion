#!/bin/bash
# Kaggle one-liner for Phase 0 SOL mapping test
set -e
pip install lightgbm scipy requests -q
git clone https://github.com/daviddnk13/Orion.git /tmp/orion
cp /kaggle/input/orion-model/model_v20_6_1.pkl /tmp/orion/
cd /tmp/orion && python3 sol_phase0_mappings.py
