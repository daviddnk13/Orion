# risk_engine.py — Orion V20.3 Risk Modulation Layer
# Input:  prob_high (float or np.array) — raw LightGBM score
# Output: position_scale (float or np.array) — multiplier for position sizing

import numpy as np
from sklearn.metrics import precision_recall_curve


class RiskEngine:
    def __init__(self, sensitivity=1.0, max_reduction=0.7, target_recall=0.70):
        """
        Args:
            sensitivity: float [0.5, 2.0] — how aggressively to scale down
            max_reduction: float [0.3, 0.8] — maximum position reduction
            target_recall: float [0.50, 0.90] — minimum recall constraint
                           for threshold optimization
        """
        self.sensitivity = np.clip(sensitivity, 0.5, 2.0)
        self.max_reduction = np.clip(max_reduction, 0.3, 0.8)
        self.target_recall = target_recall
        self.threshold = None  # set by fit()

    def fit(self, y_true, y_prob):
        """
        Find optimal threshold on TRAINING data.

        Logic:
        1. Compute precision-recall curve
        2. Find all thresholds where recall >= target_recall
        3. Among those, pick the one with highest precision
        4. If no threshold achieves target_recall, fallback to best F1

        MUST be called with training data only. Never test data.
        """
        # Ensure numpy arrays
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Guard: if only one class present, fallback to 0.5
        if len(np.unique(y_true)) < 2:
            self.threshold = 0.5
            return self

        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        # The last precision and recall values (1.0 and 0.0) do not have a corresponding threshold
        # Remove them to align with thresholds
        precision = precision[:-1]
        recall = recall[:-1]

        # Find thresholds meeting recall constraint
        candidate_indices = np.where(recall >= self.target_recall)[0]

        if len(candidate_indices) > 0:
            # Among those, choose the one with highest precision
            best_idx = candidate_indices[np.argmax(precision[candidate_indices])]
            self.threshold = thresholds[best_idx]
        else:
            # Fallback: best F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            self.threshold = thresholds[best_idx]

        return self

    def predict_scale(self, prob_high):
        """
        Convert raw P(HIGH) score to position scale.

        Formula:
            position_scale = 1.0 - clip(prob_high * sensitivity, 0, max_reduction)

        Returns: float or np.array in [1.0 - max_reduction, 1.0]
        """
        reduction = np.clip(prob_high * self.sensitivity, 0, self.max_reduction)
        position_scale = 1.0 - reduction
        return position_scale

    def predict_signal(self, prob_high):
        """
        Binary signal: is HIGH vol predicted?
        Uses threshold from fit().

        Returns: bool or np.array of bool
        """
        if self.threshold is None:
            raise RuntimeError("RiskEngine must be fit before calling predict_signal")
        return np.asarray(prob_high) >= self.threshold
