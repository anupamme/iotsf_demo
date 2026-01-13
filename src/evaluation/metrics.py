"""
IDS Evaluation Metrics

Comprehensive metrics computation for evaluating IDS performance.
"""

import numpy as np
from typing import Dict, Optional, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


class IDSMetrics:
    """
    Comprehensive metrics for IDS evaluation.

    Provides static methods for computing standard classification metrics
    (accuracy, precision, recall, F1) as well as IDS-specific metrics
    (false positive rate, detection rate).
    """

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute all metrics at once.

        Args:
            y_true: True labels (0=benign, 1=attack) of shape (n_samples,)
            y_pred: Predicted labels (0=benign, 1=attack) of shape (n_samples,)
            y_scores: Optional probability scores for ROC-AUC

        Returns:
            Dictionary containing all computed metrics:
            - accuracy: Overall accuracy
            - precision: Precision (attack prediction accuracy)
            - recall: Recall/Detection Rate (% of attacks detected)
            - f1: F1-score (harmonic mean of precision and recall)
            - confusion_matrix: 2x2 confusion matrix [[TN, FP], [FN, TP]]
            - true_negatives: Count of TN
            - false_positives: Count of FP
            - false_negatives: Count of FN (missed attacks)
            - true_positives: Count of TP (detected attacks)
            - false_positive_rate: FP / (FP + TN)
            - roc_auc: ROC-AUC score (if y_scores provided)
        """
        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        # Extract confusion matrix values
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases (e.g., all predictions are same class)
            tn = fp = fn = tp = 0
            if cm.shape == (1, 1):
                if y_true[0] == 0:  # All benign
                    tn = cm[0, 0]
                else:  # All attacks
                    tp = cm[0, 0]

        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        # False positive rate (critical for IDS)
        if (fp + tn) > 0:
            metrics['false_positive_rate'] = fp / (fp + tn)
        else:
            metrics['false_positive_rate'] = 0.0

        # ROC-AUC (if probability scores provided)
        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            except ValueError:
                # Handle case where only one class is present
                metrics['roc_auc'] = None

        return metrics

    @staticmethod
    def format_metrics_report(metrics: Dict[str, Any]) -> str:
        """
        Format metrics as human-readable string.

        Args:
            metrics: Dictionary from compute_all_metrics()

        Returns:
            Formatted string report
        """
        report = "=" * 50 + "\n"
        report += "IDS Performance Metrics\n"
        report += "=" * 50 + "\n\n"

        # Overall metrics
        report += f"Accuracy:      {metrics['accuracy']:.3f}\n"
        report += f"Precision:     {metrics['precision']:.3f}\n"
        report += f"Recall:        {metrics['recall']:.3f}\n"
        report += f"F1-Score:      {metrics['f1']:.3f}\n"
        report += f"FP Rate:       {metrics['false_positive_rate']:.3f}\n"

        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            report += f"ROC-AUC:       {metrics['roc_auc']:.3f}\n"

        report += "\n"

        # Confusion matrix
        report += "Confusion Matrix:\n"
        report += f"                Predicted\n"
        report += f"                Benign  Attack\n"
        report += f"Actual Benign   {metrics['true_negatives']:6d}  {metrics['false_positives']:6d}\n"
        report += f"       Attack   {metrics['false_negatives']:6d}  {metrics['true_positives']:6d}\n"

        report += "\n"
        report += "=" * 50 + "\n"

        return report

    @staticmethod
    def get_per_method_metrics(
        methods_dict: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple IDS methods side-by-side.

        Args:
            methods_dict: Dictionary mapping method name to IDS instance
            X_test: Test data of shape (n_samples, seq_length, feature_dim)
            y_test: True labels of shape (n_samples,)

        Returns:
            Dictionary mapping method name to its metrics dictionary

        Example:
            >>> methods = {
            ...     'Threshold': ThresholdIDS(),
            ...     'Statistical': StatisticalIDS()
            ... }
            >>> results = IDSMetrics.get_per_method_metrics(methods, X_test, y_test)
            >>> results['Threshold']['f1']
            0.823
        """
        results = {}

        for name, method in methods_dict.items():
            # Get predictions and scores
            y_pred = method.predict(X_test)
            y_scores = method.predict_proba(X_test)

            # Compute metrics
            results[name] = IDSMetrics.compute_all_metrics(
                y_test, y_pred, y_scores
            )

        return results

    @staticmethod
    def compare_methods_summary(
        results: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Create a summary comparison table of multiple methods.

        Args:
            results: Dictionary from get_per_method_metrics()

        Returns:
            Formatted comparison table string
        """
        report = "=" * 80 + "\n"
        report += "Method Comparison\n"
        report += "=" * 80 + "\n\n"

        # Header
        report += f"{'Method':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
        report += f"{'F1':>10} {'FPR':>10}\n"
        report += "-" * 80 + "\n"

        # Rows
        for method_name, metrics in results.items():
            report += f"{method_name:<15} "
            report += f"{metrics['accuracy']:>10.3f} "
            report += f"{metrics['precision']:>10.3f} "
            report += f"{metrics['recall']:>10.3f} "
            report += f"{metrics['f1']:>10.3f} "
            report += f"{metrics['false_positive_rate']:>10.3f}\n"

        report += "=" * 80 + "\n"

        return report
