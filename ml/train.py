# ml/train.py
"""
Train a baseline Logistic Regression PD model and write deployable artifacts.

Phase 2 goals:
- Train baseline PD model from a reproducible dataset snapshot.
- Save ONE sklearn Pipeline artifact so backend inference can call predict_proba directly.
- Write metadata.json so backend can expose:
  - GET /v1/model/version
  - GET /v1/metrics

Assumptions:
- You have a CSV snapshot with a binary target column (default=0/1 by default).
- feature_schema.json defines:
  - features: numeric inputs
  - segments: categorical fields like channel/region/product (optional)

Usage:
python ml/train.py \
  --data ml/data/processed/train.csv \
  --schema backend/app/ml/feature_schema.json \
  --out backend/app/ml/artifacts \
  --version v0.2.0 \
  --seed 42 \
  --target default \
  --fixed_fpr 0.05
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- Make evaluate.py import work even if ml/ is NOT a Python package ----
# This ensures `python ml/train.py ...` works without needing __init__.py
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

from evaluate import compute_metrics  # noqa: E402


def read_schema(schema_path: Path) -> Tuple[List[str], List[str]]:
    """
    Parse feature_schema.json and return:
    - numeric_features: those in "features" with numeric types
    - segment_features: keys of "segments" dict (categorical)
    """
    raw = json.loads(schema_path.read_text(encoding="utf-8"))

    numeric_features: List[str] = []
    for f in raw.get("features", []):
        ftype = str(f.get("type", "float")).lower()
        if ftype in ("int", "integer", "float", "number", "double"):
            numeric_features.append(f["name"])

    segments = raw.get("segments", {}) or {}
    segment_features = list(segments.keys())
    return numeric_features, segment_features


def sha256_file(path: Path) -> str:
    """Hash schema to detect mismatch between artifact and contract."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str, help="Training CSV snapshot path")
    parser.add_argument("--schema", required=True, type=str, help="feature_schema.json path")
    parser.add_argument("--out", required=True, type=str, help="Artifacts output directory")
    parser.add_argument("--version", required=True, type=str, help="Model version string, e.g. v0.2.0")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for split/reproducibility")
    parser.add_argument("--target", default="default", type=str, help="Target column name in CSV (0/1)")
    parser.add_argument("--fixed_fpr", default=None, type=float, help="Optional fixed FPR for recall metric, e.g. 0.05")
    parser.add_argument("--approve_threshold", default=0.30, type=float, help="Default approve threshold")
    parser.add_argument("--reject_threshold", default=0.40, type=float, help="Default reject threshold")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    schema_path = Path(args.schema).resolve()
    out_dir = Path(args.out).resolve()
    ensure_dir(out_dir)

    # ---- Load data snapshot ----
    df = pd.read_csv(data_path)

    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found in {data_path}. "
            f"Available columns: {list(df.columns)}"
        )

    y = df[args.target].astype(int).to_numpy()

    # ---- Determine feature columns from schema ----
    numeric_features, segment_features = read_schema(schema_path)

    # Keep only columns that actually exist in df
    numeric_features = [c for c in numeric_features if c in df.columns]
    segment_features = [c for c in segment_features if c in df.columns]

    if not numeric_features:
        raise ValueError(
            "No numeric feature columns found in CSV according to schema. "
            "Check feature_schema.json 'features' list and your CSV columns."
        )

    X = df[numeric_features + segment_features].copy()

    # ---- Build preprocessing ----
    # Numeric: median impute + standardize
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical segments: most_frequent impute + one-hot encode
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, segment_features),
        ],
        remainder="drop",
    )

    # ---- Baseline model ----
    # Logistic Regression is a classic baseline for credit scoring / PD modeling.
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",  # helpful when default is minority
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # ---- Train/Val/Test split (stratified) ----
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=args.seed,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=0.50,
        random_state=args.seed,
        stratify=y_tmp,
    )

    # ---- Fit ----
    pipeline.fit(X_train, y_train)

    # ---- Predict PD and evaluate ----
    val_pd = pipeline.predict_proba(X_val)[:, 1]
    test_pd = pipeline.predict_proba(X_test)[:, 1]

    metrics_val = compute_metrics(y_val, val_pd, fixed_fpr=args.fixed_fpr)
    metrics_test = compute_metrics(y_test, test_pd, fixed_fpr=args.fixed_fpr)

    # ---- Write artifacts ----
    model_file = out_dir / "model.joblib"
    metadata_file = out_dir / "metadata.json"

    # Save the full sklearn pipeline so backend can do:
    # pipeline = joblib.load(model.joblib)
    # pd = pipeline.predict_proba(df_like)[:, 1]
    joblib.dump(pipeline, model_file)

    # Metadata (backend endpoints read this)
    metadata: Dict[str, Any] = {
        "model_name": "logreg_baseline",
        "version": args.version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_schema_sha256": sha256_file(schema_path),

        # Inputs used by the model (for transparency + inference contract checks)
        "features": numeric_features,
        "segments": segment_features,

        # Store metrics by split
        "metrics": {
            "val": metrics_val,
            "test": metrics_test,
        },

        # Dataset snapshot info for reproducibility
        "data_info": {
            "n_rows": int(len(df)),
            "default_rate": float(df[args.target].mean()),
            "split": {"train": 0.70, "val": 0.15, "test": 0.15},
            "seed": int(args.seed),
            "data_path": str(data_path),
            "target": args.target,
        },

        # Default thresholds live here so Phase 3/7 can reuse them
        "thresholds": {
            "approve": float(args.approve_threshold),
            "reject": float(args.reject_threshold),
        },

        "artifact_files": {
            "model": model_file.name,
            "metadata": metadata_file.name,
        },
    }

    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[train] model saved to: {model_file}")
    print(f"[train] metadata saved to: {metadata_file}")
    print(f"[train] val metrics: {metrics_val}")
    print(f"[train] test metrics: {metrics_test}")


if __name__ == "__main__":
    main()