# ============================================================
# temporal.py — QVG v1.1: Temporal integrity validation
# Detects train/test leakage, embargo violations, non-monotonic index
# ============================================================

import numpy as np


class TemporalCheckResult:
    def __init__(self):
        self.passed = True
        self.errors = []

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = ["TemporalCheck: {}".format(status)]
        for e in self.errors:
            lines.append("  - " + e)
        return "\n".join(lines)


def run_temporal_checks(df, horizon=6):
    """Validate temporal integrity of predictions DataFrame.
    
    Args:
        df: DataFrame with columns y_true, y_pred, fold.
            Optionally: timestamp, train_end_idx, test_start_idx
        horizon: prediction horizon in bars
    """
    result = TemporalCheckResult()

    # 1. Monotonic index (rows should be time-ordered)
    if 'timestamp' in df.columns:
        ts = df['timestamp'].values
        if not np.all(ts[1:] >= ts[:-1]):
            result.fail("Timestamps are NOT monotonically increasing")
    else:
        # Without timestamps, check index monotonicity
        if not df.index.is_monotonic_increasing:
            result.fail(
                "Index is not monotonically increasing "
                "(no timestamp column available)")

    # 2. Embargo check between folds
    folds = sorted(df['fold'].unique())
    if len(folds) >= 2:
        for i in range(len(folds) - 1):
            f_curr = df[df['fold'] == folds[i]]
            f_next = df[df['fold'] == folds[i + 1]]

            curr_end = f_curr.index.max()
            next_start = f_next.index.min()
            gap = next_start - curr_end

            if gap < horizon:
                result.fail(
                    "Embargo violation between fold {} and {}: "
                    "gap={} bars, required>={} (horizon)".format(
                        folds[i], folds[i + 1], gap, horizon))

    # 3. Train/test index validation (if columns present)
    if 'train_end_idx' in df.columns and 'test_start_idx' in df.columns:
        for fold in folds:
            f_data = df[df['fold'] == fold]
            train_end = f_data['train_end_idx'].iloc[0]
            test_start = f_data['test_start_idx'].iloc[0]
            gap = test_start - train_end

            if gap < horizon:
                result.fail(
                    "Fold {}: train_end={} test_start={} "
                    "gap={} < horizon={}".format(
                        fold, train_end, test_start, gap, horizon))

            if test_start <= train_end:
                result.fail(
                    "Fold {}: test overlaps train! "
                    "train_end={} test_start={}".format(
                        fold, train_end, test_start))

    # 4. No duplicate indices within a fold
    for fold in folds:
        f_data = df[df['fold'] == fold]
        n_unique = f_data.index.nunique()
        if n_unique < len(f_data):
            result.fail(
                "Fold {}: {} duplicate indices detected".format(
                    fold, len(f_data) - n_unique))

    return result
