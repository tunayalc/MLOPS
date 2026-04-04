import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import mlflow
from fairlearn.metrics import MetricFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_SENSITIVE_FEATURE = "sex"


def _compute_regression_fairness(
    y_true: pd.Series,
    y_pred: np.ndarray,
    sensitive: pd.Series,
    *,
    sensitive_name: str,
) -> Dict[str, Any]:
    """Compute group-wise regression error metrics using Fairlearn."""
    metrics = {
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "r2": r2_score,
    }

    frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    by_group_df = frame.by_group
    by_group: Dict[str, Dict[str, float]] = {}
    for group_value in by_group_df.index:
        group_metrics: Dict[str, float] = {}
        for metric_name in metrics:
            group_metrics[metric_name] = float(
                by_group_df.loc[group_value, metric_name]
            )
        by_group[str(group_value)] = group_metrics

    overall = {
        name: float(func(y_true, y_pred)) for name, func in metrics.items()
    }

    disparities: Dict[str, float] = {}
    for metric_name in metrics:
        group_values = [group[metric_name] for group in by_group.values()]
        if group_values:
            disparities[metric_name] = float(
                max(group_values) - min(group_values)
            )

    return {
        "sensitive_feature": sensitive_name,
        "overall": overall,
        "by_group": by_group,
        "disparities": disparities,
    }


def log_fairness_report(
    *,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_predictions: np.ndarray,
    run_id: str,
    output_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute a Fairlearn-based fairness report for the regression model and
    persist it both locally and as an MLflow artifact.
    """
    if output_dir is None:
        output_dir = Path("fairness_reports")

    if DEFAULT_SENSITIVE_FEATURE not in X_test.columns:
        # Dataset does not expose the default sensitive feature; skip gracefully.
        return None

    sensitive = X_test[DEFAULT_SENSITIVE_FEATURE]
    report = _compute_regression_fairness(
        y_true=y_test,
        y_pred=test_predictions,
        sensitive=sensitive,
        sensitive_name=DEFAULT_SENSITIVE_FEATURE,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{run_id}_fairlearn_fairness_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

    mlflow.log_dict(
        report,
        artifact_file="fairness/fairlearn_fairness_report.json",
    )

    return report

