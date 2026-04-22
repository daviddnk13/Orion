# ============================================================
# quant_gate — Quant Validation Gate v1.1c
# Hard gate: failure = STOP + BLOCK
# v1.1c FIXES: Spearman unified in significance, horizon
#   passthrough to block_permutation_test, block_size capped,
#   overfitting check wired in validator
# ============================================================

__version__ = "1.1.1c"

from .validator import validate, load_predictions
