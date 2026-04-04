import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _save_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    destination: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="white", linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _save_residual_scatter(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    destination: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolor="white", linewidth=0.5)
    ax.axhline(0.0, linestyle="--", color="red")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _save_residual_hist(
    residuals: np.ndarray,
    destination: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=20, alpha=0.75, color="#4472C4", edgecolor="black")
    ax.axvline(0.0, linestyle="--", color="red")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _collect_feature_importance(
    pipeline: Pipeline,
    top_n: int = 20,
) -> Optional[pd.DataFrame]:
    if "regressor" not in pipeline.named_steps:
        return None

    regressor = pipeline.named_steps["regressor"]
    if not hasattr(regressor, "feature_importances_"):
        return None

    if "preprocessor" not in pipeline.named_steps:
        return None

    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        return None

    importances = regressor.feature_importances_
    if len(feature_names) != len(importances):
        return None

    df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    return df.head(top_n).reset_index(drop=True)


def _save_feature_importance_plot(
    df: pd.DataFrame,
    destination: Path,
    title: str = "Top Feature Importances",
) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="#2F5597")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def log_regression_artifacts(
    *,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_predictions: np.ndarray,
    test_predictions: np.ndarray,
    metrics: Dict[str, float],
    artifact_path: str = "analysis",
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """Generate diagnostic plots and log them to MLflow."""
    residuals_train = (y_train.values - train_predictions).astype(float)
    residuals_test = (y_test.values - test_predictions).astype(float)

    predictions_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "set": "train",
                    "actual": y_train.values,
                    "prediction": train_predictions,
                    "residual": residuals_train,
                }
            ),
            pd.DataFrame(
                {
                    "set": "test",
                    "actual": y_test.values,
                    "prediction": test_predictions,
                    "residual": residuals_test,
                }
            ),
        ],
        ignore_index=True,
    )

    summary = {
        "metrics": {key: float(value) for key, value in metrics.items()},
        "train_residuals": {
            "mean": float(np.mean(residuals_train)),
            "std": float(np.std(residuals_train)),
            "min": float(np.min(residuals_train)),
            "max": float(np.max(residuals_train)),
        },
        "test_residuals": {
            "mean": float(np.mean(residuals_test)),
            "std": float(np.std(residuals_test)),
            "min": float(np.min(residuals_test)),
            "max": float(np.max(residuals_test)),
        },
    }

    top_features = _collect_feature_importance(pipeline)

    if tags:
        mlflow.set_tags(tags)

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        predictions_path = tmp_path / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)

        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        _save_actual_vs_predicted(
            y_test,
            test_predictions,
            tmp_path / "actual_vs_predicted_test.png",
            "Actual vs Predicted (Test)",
        )

        _save_residual_scatter(
            test_predictions,
            residuals_test,
            tmp_path / "residuals_vs_prediction_test.png",
            "Residuals vs Prediction (Test)",
        )

        _save_residual_hist(
            residuals_test,
            tmp_path / "residual_distribution_test.png",
            "Residual Distribution (Test)",
        )

        if top_features is not None and not top_features.empty:
            feature_table_path = tmp_path / "top_feature_importances.csv"
            top_features.to_csv(feature_table_path, index=False)
            _save_feature_importance_plot(
                top_features,
                tmp_path / "top_feature_importances.png",
            )

        mlflow.log_artifacts(tmp_path, artifact_path=artifact_path)
