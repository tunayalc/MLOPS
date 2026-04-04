import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from experiment_store import connect, insert_dataset_split, insert_experiment
from fairness_reporting import log_fairness_report
from mlsecops_guardrails import (
    assess_noise_robustness,
    build_security_report,
    capture_data_lineage,
    ensure_manifest,
    evaluate_generalization_risk,
    write_security_report,
)
from regression_reporting import log_regression_artifacts
from pipeline_utils import (
    DATA_URL,
    TARGET_COLUMN,
    build_pipeline,
    load_student_performance_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a student performance regression model."
    )
    parser.add_argument(
        "--db-path",
        default="experiments.db",
        help="SQLite database file to store experiment metadata.",
    )
    parser.add_argument(
        "--experiment-name",
        default="student-performance-regression",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-name",
        default="baseline-random-forest",
        help="Optional MLflow run name.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for splitting and the model.",
    )
    parser.add_argument(
        "--use-mlflow-sqlite",
        action="store_true",
        help="Use a SQLite backend store for MLflow tracking.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Override tracking URI. Takes precedence over --use-mlflow-sqlite.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Optional free-form notes stored alongside the experiment metadata.",
    )
    parser.add_argument(
        "--security-manifest",
        default="mlsecops_manifest.json",
        help="Path to the MLSecOps dataset lineage manifest.",
    )
    parser.add_argument(
        "--enforce-lineage",
        action="store_true",
        help="Fail the run if dataset hash differs from the manifest.",
    )
    parser.add_argument(
        "--robustness-noise-scale",
        type=float,
        default=0.15,
        help="Noise scale applied during adversarial robustness sampling.",
    )
    parser.add_argument(
        "--robustness-samples",
        type=int,
        default=5,
        help="Number of noisy samples used for robustness estimation.",
    )
    parser.add_argument(
        "--generalization-threshold",
        type=float,
        default=0.15,
        help="Relative fark bu eşiği aşarsa OWASP ML03-ML04 için uyarı üret.",
    )
    parser.add_argument(
        "--security-report-dir",
        default="mlsecops_reports",
        help="Directory that will store MLSecOps audit reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    data_frame = load_student_performance_dataset(DATA_URL)
    if TARGET_COLUMN not in data_frame.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    lineage_record = capture_data_lineage(data_frame, source=DATA_URL)
    lineage_validation = ensure_manifest(
        lineage_record,
        Path(args.security_manifest),
        enforce=args.enforce_lineage,
    )

    features = data_frame.drop(columns=[TARGET_COLUMN])
    target = data_frame[TARGET_COLUMN]

    numeric_features = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = features.select_dtypes(include=["object", "category"]).columns.tolist()

    hyperparameters = {
        "n_estimators": 200,
        "max_depth": 8,
        "random_state": args.random_state,
        "n_jobs": -1,
    }

    pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        hyperparameters=hyperparameters,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    tracking_uri = args.mlflow_tracking_uri
    if tracking_uri is None and args.use_mlflow_sqlite:
        tracking_uri = f"sqlite:///{db_path.with_suffix('.mlflow.db')}"
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        pipeline.fit(X_train, y_train)
        test_predictions = pipeline.predict(X_test)
        train_predictions = pipeline.predict(X_train)

        test_mse = mean_squared_error(y_test, test_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        train_mse = mean_squared_error(y_train, train_predictions)
        train_mae = mean_absolute_error(y_train, train_predictions)
        train_r2 = r2_score(y_train, train_predictions)

        metrics = {
            "train_mse": train_mse,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_r2": test_r2,
        }
        robustness = assess_noise_robustness(
            pipeline,
            features=X_test,
            target=y_test,
            numeric_features=numeric_features,
            baseline_metrics=metrics,
            noise_scale=args.robustness_noise_scale,
            samples=args.robustness_samples,
        )
        generalization = evaluate_generalization_risk(
            metrics,
            threshold=args.generalization_threshold,
        )
        security_report = build_security_report(
            lineage=lineage_record,
            lineage_validation=lineage_validation,
            pipeline=pipeline,
            robustness=robustness,
            hyperparameters=hyperparameters,
            generalization=generalization,
        )
        report_path = (
            Path(args.security_report_dir) / f"{run.info.run_id}_mlsecops_report.json"
        )
        write_security_report(security_report, report_path)

        # Fairness / responsible AI audit using Fairlearn.
        fairness_report = log_fairness_report(
            X_test=X_test,
            y_test=y_test,
            test_predictions=test_predictions,
            run_id=run.info.run_id,
        )

        mlflow.log_metrics(metrics)
        gen_gap_mae = generalization["gaps"]["mae"]["relative_gap"]
        gen_gap_mse = generalization["gaps"]["mse"]["relative_gap"]
        gen_gap_r2 = generalization["gaps"]["r2"]["absolute_gap"]
        generalization_metrics = {}
        if gen_gap_mae is not None:
            generalization_metrics["generalization_gap_mae"] = gen_gap_mae
        if gen_gap_mse is not None:
            generalization_metrics["generalization_gap_mse"] = gen_gap_mse
        if gen_gap_r2 is not None:
            generalization_metrics["generalization_gap_r2"] = gen_gap_r2
        if generalization_metrics:
            mlflow.log_metrics(generalization_metrics)
        mlflow.log_params({f"model__{key}": value for key, value in hyperparameters.items()})
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("dataset_hash", lineage_record.dataset_hash)
        mlflow.log_param("generalization_status", generalization["status"])
        mlflow.log_param("generalization_threshold", args.generalization_threshold)
        mlflow.log_dict(security_report, artifact_file="security/mlsecops_report.json")
        mlflow.log_artifact(report_path, artifact_path="security")
        mlflow.log_artifact(Path(__file__).name)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        log_regression_artifacts(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_predictions=train_predictions,
            test_predictions=test_predictions,
            metrics=metrics,
            artifact_path="analysis",
            tags={"target": TARGET_COLUMN},
        )

        train_config = {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "target_column": TARGET_COLUMN,
            "feature_count": len(features.columns),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "dataset_hash": lineage_record.dataset_hash,
            "security_manifest": str(Path(args.security_manifest)),
            "security_report": str(report_path),
            "generalization_threshold": args.generalization_threshold,
        }

        with connect(str(db_path)) as conn:
            experiment_id = insert_experiment(
                conn,
                model_type="RandomForestRegressor",
                hyperparameters=hyperparameters,
                train_config=train_config,
                mlflow_run_id=run.info.run_id,
                mlflow_tracking_uri=mlflow.get_tracking_uri(),
                metrics=metrics,
                data_source=DATA_URL,
                notes=args.notes,
            )
            insert_dataset_split(
                conn,
                experiment_id=experiment_id,
                split="train",
                features_rows=X_train.reset_index(drop=True).to_dict(orient="records"),
                target_values=y_train.reset_index(drop=True).tolist(),
            )
            insert_dataset_split(
                conn,
                experiment_id=experiment_id,
                split="test",
                features_rows=X_test.reset_index(drop=True).to_dict(orient="records"),
                target_values=y_test.reset_index(drop=True).tolist(),
            )

    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "mlflow_run_id": run.info.run_id,
                "metrics": metrics,
                "tracking_uri": mlflow.get_tracking_uri(),
                "db_path": str(db_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
