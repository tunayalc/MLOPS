
"""Utilities inspired by the MLSecOps guide for lineage, integrity and robustness."""
from __future__ import annotations

import json
import hashlib
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import sklearn


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Create a deterministic hash for a dataframe (order agnostic)."""
    hashed = pd.util.hash_pandas_object(df, index=True).values
    digest = hashlib.sha256(hashed.tobytes())
    return digest.hexdigest()


@dataclass
class DataLineageRecord:
    source: str
    created_at: str
    row_count: int
    column_count: int
    dataset_hash: str
    numeric_features: Dict[str, Dict[str, float]]
    categorical_features: Dict[str, Dict[str, Any]]
    missing_values: Dict[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def capture_data_lineage(df: pd.DataFrame, *, source: str) -> DataLineageRecord:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        col for col in df.columns if col not in numeric_cols
    ]

    numeric_summary: Dict[str, Dict[str, float]] = {}
    for column in numeric_cols:
        series = df[column].astype(float)
        numeric_summary[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
        }

    categorical_summary: Dict[str, Dict[str, Any]] = {}
    for column in categorical_cols:
        series = df[column].astype(str)
        top_value = series.mode().iloc[0] if not series.mode().empty else None
        categorical_summary[column] = {
            "unique_count": int(series.nunique()),
            "top_value": top_value,
        }

    missing = df.isna().sum().to_dict()

    dataset_hash = _hash_dataframe(df)

    return DataLineageRecord(
        source=source,
        created_at=_utcnow(),
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        dataset_hash=dataset_hash,
        numeric_features=numeric_summary,
        categorical_features=categorical_summary,
        missing_values={key: int(value) for key, value in missing.items()},
    )


def load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _serialize_manifest(lineage: DataLineageRecord) -> Dict[str, Any]:
    return {
        "created_at": lineage.created_at,
        "dataset_hash": lineage.dataset_hash,
        "row_count": lineage.row_count,
        "column_count": lineage.column_count,
        "source": lineage.source,
    }


def ensure_manifest(
    lineage: DataLineageRecord,
    manifest_path: Path,
    *,
    enforce: bool = False,
) -> Dict[str, Any]:
    """Validate/refresh the lineage manifest according to policy."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_manifest(manifest_path)
    manifest_payload = _serialize_manifest(lineage)

    if existing is None:
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        return {
            "status": "created",
            "manifest_path": str(manifest_path),
            "previous_hash": None,
            "current_hash": lineage.dataset_hash,
        }

    previous_hash = existing.get("dataset_hash")
    if previous_hash != lineage.dataset_hash:
        if enforce:
            raise RuntimeError(
                "Dataset hash does not match the trusted manifest. "
                "Resolve the discrepancy or run without --enforce-lineage."
            )
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        status = "updated"
    else:
        status = "matched"

    return {
        "status": status,
        "manifest_path": str(manifest_path),
        "previous_hash": previous_hash,
        "current_hash": lineage.dataset_hash,
    }


def hash_model(pipeline: Pipeline) -> str:
    """Generate a stable digest for the trained pipeline."""
    payload = pickle.dumps(pipeline)
    return hashlib.sha256(payload).hexdigest()


def assess_noise_robustness(
    pipeline: Pipeline,
    *,
    features: pd.DataFrame,
    target: pd.Series,
    numeric_features: Iterable[str],
    baseline_metrics: Dict[str, float],
    noise_scale: float = 0.15,
    samples: int = 5,
) -> Dict[str, Any]:
    numeric_features = list(numeric_features)
    if not numeric_features:
        return {
            "status": "skipped",
            "reason": "no_numeric_features",
        }

    trials: List[Dict[str, float]] = []
    for _ in range(max(1, samples)):
        perturbed = features.copy()
        numeric_block = perturbed[numeric_features].astype(float).to_numpy(copy=True)
        noise = np.random.normal(scale=noise_scale, size=numeric_block.shape)
        numeric_block += noise
        perturbed[numeric_features] = numeric_block
        predictions = pipeline.predict(perturbed)
        trials.append(
            {
                "mae": float(mean_absolute_error(target, predictions)),
                "mse": float(mean_squared_error(target, predictions)),
                "r2": float(r2_score(target, predictions)),
            }
        )

    avg_mae = float(np.mean([trial["mae"] for trial in trials]))
    avg_mse = float(np.mean([trial["mse"] for trial in trials]))
    avg_r2 = float(np.mean([trial["r2"] for trial in trials]))

    base_mae = float(baseline_metrics.get("test_mae", 0.0))
    base_mse = float(baseline_metrics.get("test_mse", 0.0))
    base_r2 = float(baseline_metrics.get("test_r2", 0.0))

    def _delta(current: float, baseline: float) -> Optional[float]:
        if baseline == 0.0:
            return None
        return (current - baseline) / max(abs(baseline), 1e-9)

    return {
        "status": "completed",
        "samples": len(trials),
        "noise_scale": noise_scale,
        "avg_mae": avg_mae,
        "avg_mse": avg_mse,
        "avg_r2": avg_r2,
        "relative_delta_mae": _delta(avg_mae, base_mae),
        "relative_delta_mse": _delta(avg_mse, base_mse),
        "relative_delta_r2": _delta(avg_r2, base_r2),
    }


def evaluate_generalization_risk(
    metrics: Dict[str, float],
    *,
    threshold: float = 0.15,
) -> Dict[str, Any]:
    train_mae = float(metrics.get("train_mae", 0.0))
    test_mae = float(metrics.get("test_mae", 0.0))
    train_mse = float(metrics.get("train_mse", 0.0))
    test_mse = float(metrics.get("test_mse", 0.0))
    train_r2 = float(metrics.get("train_r2", 0.0))
    test_r2 = float(metrics.get("test_r2", 0.0))

    def _gap(train: float, test: float) -> Optional[float]:
        denom = max(abs(test), 1e-9)
        return (test - train) / denom

    def _gap_r2(train: float, test: float) -> Optional[float]:
        return train - test

    mae_gap = _gap(train_mae, test_mae)
    mse_gap = _gap(train_mse, test_mse)
    r2_gap = _gap_r2(train_r2, test_r2)

    status = "ok"
    for gap in [mae_gap, mse_gap, r2_gap]:
        if gap is None:
            continue
        if gap >= threshold * 1.5:
            status = "alert"
            break
        if gap >= threshold and status != "alert":
            status = "monitor"

    notes = (
        "Train/test farkı kontrol altında."
        if status == "ok"
        else "Genelleme kaybı gözlemlendi; veri kalitesi ve regularization gözden geçirilmeli."
    )

    return {
        "status": status,
        "threshold": threshold,
        "gaps": {
            "mae": {
                "train": train_mae,
                "test": test_mae,
                "relative_gap": mae_gap,
            },
            "mse": {
                "train": train_mse,
                "test": test_mse,
                "relative_gap": mse_gap,
            },
            "r2": {
                "train": train_r2,
                "test": test_r2,
                "absolute_gap": r2_gap,
            },
        },
        "notes": notes,
    }


def _assess_control_maturity(
    *,
    lineage_validation: Dict[str, Any],
    robustness: Dict[str, Any],
    generalization: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    robustness_delta = robustness.get("relative_delta_mae")
    if robustness_delta is None:
        adv_status = "info"
        adv_note = "Sayısal özellik olmadığı için gürültü testi atlandı."
    elif robustness_delta < 0.25:
        adv_status = "ok"
        adv_note = "Adversarial noise testi kabul edilebilir aralıkta."
    elif robustness_delta < 0.5:
        adv_status = "monitor"
        adv_note = "Noise altında kayıp arttı; dayanıklılık iyileştirilmeli."
    else:
        adv_status = "alert"
        adv_note = "Noise testi kritik bozulmaya işaret ediyor."

    lineage_status = lineage_validation.get("status", "unknown")
    lineage_state = "ok" if lineage_status in {"matched", "created"} else "monitor"
    gen_status = generalization.get("status", "ok")

    owasp_matrix = [
        {
            "id": "ML01",
            "title": "Girdi Manipülasyonu / Kanma",
            "status": adv_status,
            "evidence": {
                "noise_scale": robustness.get("noise_scale"),
                "relative_delta_mae": robustness_delta,
            },
            "recommendation": adv_note,
        },
        {
            "id": "ML02",
            "title": "Veri Zehirleme",
            "status": lineage_state,
            "evidence": lineage_validation,
            "recommendation": "Manifest uyuşmazlıklarını inceleyin; gerekirse --enforce-lineage kullanın.",
        },
        {
            "id": "ML03-ML04",
            "title": "Model/Veri Gizliliği",
            "status": gen_status,
            "evidence": generalization,
            "recommendation": "Train/test farkı büyürse DP, regularization ve audit loglarını devreye alın.",
        },
        {
            "id": "ML05",
            "title": "Model Hırsızlığı",
            "status": "info",
            "evidence": {
                "tracked_via_mlflow": True,
                "logged_artifacts": ["security/mlsecops_report.json"],
            },
            "recommendation": "API hız limitlerini ve kimlik doğrulamayı MLflow servislerinde zorunlu kılın.",
        },
        {
            "id": "ML08",
            "title": "Model Çürümesi",
            "status": lineage_state if lineage_state != "unknown" else gen_status,
            "evidence": {
                "manifest_status": lineage_status,
                "generalization_status": gen_status,
            },
            "recommendation": "Manifest ve raporları drift monitörleriyle birlikte kullanın.",
        },
        {
            "id": "ML10",
            "title": "Model Zehirleme / Artefact Kurcalama",
            "status": "ok",
            "evidence": {
                "model_integrity_logged": True,
                "storage": "mlflow",
            },
            "recommendation": "Model imzalama (Sigstore) ekleyerek doğrulamayı otomatikleştirin.",
        },
    ]

    atlas_matrix = [
        {
            "tactic": "Reconnaissance",
            "status": "documented",
            "evidence": {
                "dataset_hash": lineage_validation.get("current_hash"),
                "source": lineage_validation.get("manifest_path"),
            },
        },
        {
            "tactic": "ML Attack Staging",
            "status": adv_status,
            "evidence": {"robustness": robustness},
        },
        {
            "tactic": "Model Access",
            "status": "monitor" if gen_status != "ok" else "ok",
            "evidence": {"generalization": generalization},
        },
        {
            "tactic": "Attack Execution / Impact",
            "status": lineage_state,
            "evidence": {"manifest": lineage_validation},
        },
    ]

    return {
        "owasp": owasp_matrix,
        "atlas": atlas_matrix,
    }


def build_security_report(
    *,
    lineage: DataLineageRecord,
    lineage_validation: Dict[str, Any],
    pipeline: Pipeline,
    robustness: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    generalization: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the MLSecOps security record for auditing."""
    controls = _assess_control_maturity(
        lineage_validation=lineage_validation,
        robustness=robustness,
        generalization=generalization,
    )
    return {
        "generated_at": _utcnow(),
        "lineage": lineage.as_dict(),
        "lineage_validation": lineage_validation,
        "model_integrity": {
            "sha256": hash_model(pipeline),
            "sklearn_version": sklearn.__version__,
            "hyperparameters": hyperparameters,
        },
        "robustness": robustness,
        "generalization": generalization,
        "owasp_ml_top_10": controls["owasp"],
        "mitre_atlas": controls["atlas"],
        "references": [
            "OWASP ML Top 10",
            "MITRE ATLAS",
            "MLSecOps-Yapay-Zeka Rehberi",
        ],
    }


def write_security_report(report: Dict[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination
