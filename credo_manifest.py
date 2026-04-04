import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from experiment_store import connect, fetch_latest_experiment


def _latest_match(pattern: str) -> Optional[Path]:
    candidates = sorted(
        Path(".").glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> None:
    db_path = Path("experiments.db")

    with connect(str(db_path)) as conn:
        experiment = fetch_latest_experiment(conn)

    mlsecops_path = _latest_match("mlsecops_reports/*_mlsecops_report.json")
    fairness_path = _latest_match(
        "fairness_reports/*_fairlearn_fairness_report.json"
    )
    giskard_path = _latest_match("giskard_reports/*.json")
    sbom_path = Path("cyclonedx_sbom.json")

    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "id": experiment["id"],
            "model_type": experiment["model_type"],
            "created_at": experiment["created_at"],
            "metrics": experiment["metrics"],
            "mlflow_run_id": experiment["mlflow_run_id"],
            "mlflow_tracking_uri": experiment["mlflow_tracking_uri"],
            "data_source": experiment["data_source"],
        },
        "artifacts": {},
        "notes": "Aggregated technical evidence for AI governance / Credo AI-style assessment.",
    }

    if mlsecops_path and mlsecops_path.exists():
        manifest["artifacts"]["mlsecops_report"] = {
            "path": str(mlsecops_path),
            "content": json.loads(mlsecops_path.read_text(encoding="utf-8")),
        }

    if fairness_path and fairness_path.exists():
        manifest["artifacts"]["fairlearn_fairness"] = {
            "path": str(fairness_path),
            "content": json.loads(fairness_path.read_text(encoding="utf-8")),
        }

    if giskard_path and giskard_path.exists():
        manifest["artifacts"]["giskard_scan"] = {
            "path": str(giskard_path),
            "content": json.loads(giskard_path.read_text(encoding="utf-8")),
        }

    if sbom_path.exists():
        manifest["artifacts"]["cyclonedx_sbom"] = {
            "path": str(sbom_path),
        }

    destination = Path("credo_ai_manifest.json")
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_path": str(destination)}, indent=2))


if __name__ == "__main__":
    main()

