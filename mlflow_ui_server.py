# mlflow_ui_server.py
from pathlib import Path
from waitress import serve
from mlflow.server import get_app

backend = "sqlite:///{}".format(Path("experiments.mlflow.db").resolve())
artifact_root = Path("mlruns").resolve().as_uri()

app = get_app({"backend_store_uri": backend, "default_artifact_root": artifact_root})
serve(app, host="127.0.0.1", port=5000)
