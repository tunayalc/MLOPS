import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from giskard import Dataset, Model, scan

from experiment_store import connect, fetch_latest_experiment
from pipeline_utils import DATA_URL, TARGET_COLUMN, load_student_performance_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Giskard scan for the latest tabular experiment.",
    )
    parser.add_argument(
        "--db-path",
        default="experiments.db",
        help="SQLite database file that contains experiment metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default="giskard_reports",
        help="Directory to store Giskard scan reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)

    with connect(str(db_path)) as conn:
        record = fetch_latest_experiment(conn)

    tracking_uri = record.get("mlflow_tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    run_id = record.get("mlflow_run_id")
    if not run_id:
        raise RuntimeError(
            "Latest experiment does not contain an MLflow run id."
        )

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    data_source = record.get("data_source") or DATA_URL
    dataframe = load_student_performance_dataset(data_source)
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    feature_names = [col for col in dataframe.columns if col != TARGET_COLUMN]

    giskard_model = Model(
        model=model,
        model_type="regression",
        name=f"student_regression_experiment_{record['id']}",
        feature_names=feature_names,
    )
    dataset = Dataset(
        df=dataframe,
        target=TARGET_COLUMN,
        name="student_performance_dataset",
    )

    report = scan(giskard_model, dataset, verbose=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"giskard_scan_experiment_{record['id']}.html"
    json_path = output_dir / f"giskard_scan_experiment_{record['id']}.json"

    report.to_html(str(html_path))
    report.to_json(str(json_path))

    # Log a concise summary into the associated MLflow run as well.
    report.to_mlflow(mlflow_run_id=run_id)

    summary = {
        "experiment_id": record["id"],
        "mlflow_run_id": run_id,
        "html_report": str(html_path),
        "json_report": str(json_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

