"""Evaluation metrics for binary arc detection.

Computes precision, recall, F1 score, and ROC-AUC with support for
batch-wise accumulation during training/validation loops.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class ArcDetectionMetrics:
    """Accumulates predictions across batches and computes detection metrics.

    Tracks true positives, false positives, false negatives, and
    prediction probabilities for AUC computation.

    Usage:
        metrics = ArcDetectionMetrics()
        for batch in loader:
            metrics.update(preds, labels, probs)
        results = metrics.compute()
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._true_positives = 0
        self._false_positives = 0
        self._false_negatives = 0
        self._true_negatives = 0
        self._all_labels: list[int] = []
        self._all_probs: list[float] = []

    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        probs: torch.Tensor | None = None,
    ) -> None:
        """Accumulate a batch of predictions.

        Args:
            preds: Predicted class indices, shape (batch,).
            labels: Ground truth class indices, shape (batch,).
            probs: Predicted probability of the positive class, shape (batch,).
                   Required for AUC computation.
        """
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        self._true_positives += int(((preds_np == 1) & (labels_np == 1)).sum())
        self._false_positives += int(((preds_np == 1) & (labels_np == 0)).sum())
        self._false_negatives += int(((preds_np == 0) & (labels_np == 1)).sum())
        self._true_negatives += int(((preds_np == 0) & (labels_np == 0)).sum())

        self._all_labels.extend(labels_np.tolist())

        if probs is not None:
            self._all_probs.extend(probs.detach().cpu().numpy().tolist())

    def compute(self) -> dict[str, float]:
        """Compute all metrics from accumulated predictions.

        Returns:
            Dictionary with keys: precision, recall, f1, auc, accuracy.
        """
        tp = self._true_positives
        fp = self._false_positives
        fn = self._false_negatives
        tn = self._true_negatives

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        total = tp + fp + fn + tn
        accuracy = (tp + tn) / max(total, 1)

        auc = 0.0
        if self._all_probs and len(set(self._all_labels)) > 1:
            try:
                auc = roc_auc_score(
                    np.array(self._all_labels), np.array(self._all_probs)
                )
            except ValueError:
                auc = 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "accuracy": accuracy,
        }

    def summary(self) -> str:
        """Return a formatted string summary of all metrics."""
        m = self.compute()
        return (
            f"Precision: {m['precision']:.4f} | "
            f"Recall: {m['recall']:.4f} | "
            f"F1: {m['f1']:.4f} | "
            f"AUC: {m['auc']:.4f} | "
            f"Accuracy: {m['accuracy']:.4f}"
        )
