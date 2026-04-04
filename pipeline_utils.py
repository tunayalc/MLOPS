from pathlib import Path
from typing import Dict, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve
import zipfile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"
)
DATA_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
)
CACHE_DIR = Path(".data")
TARGET_COLUMN = "G3"


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR


def _download_file(url: str, destination: Path) -> None:
    urlretrieve(url, destination)


def _extract_from_zip(zip_path: Path, member: str, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as source, destination.open("wb") as target:
            target.write(source.read())


def load_student_performance_dataset(url: str = DATA_URL) -> pd.DataFrame:
    """Download (with caching and fallbacks) and return the student dataset."""
    cache_dir = _ensure_cache_dir()
    local_csv = cache_dir / "student-mat.csv"

    if local_csv.exists():
        return pd.read_csv(local_csv, sep=";")

    try:
        _download_file(url, local_csv)
        return pd.read_csv(local_csv, sep=";")
    except (HTTPError, URLError):
        # Fallback to the zip archive provided by the UCI repository.
        zip_path = cache_dir / "student.zip"
        try:
            _download_file(DATA_ZIP_URL, zip_path)
            _extract_from_zip(zip_path, "student-mat.csv", local_csv)
            return pd.read_csv(local_csv, sep=";")
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                "Unable to download the student performance dataset. "
                "You can place 'student-mat.csv' in the '.data' directory manually."
            ) from exc


def build_pipeline(
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    hyperparameters: Dict,
) -> Pipeline:
    """Create the preprocessing + RandomForestRegressor pipeline."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ]
    )

    regressor = RandomForestRegressor(**hyperparameters)
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])
