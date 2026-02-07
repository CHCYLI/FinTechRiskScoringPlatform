"""
Train a baseline Logistic Regression PD model and write deployable artifacts.

Phase 2 goals:
- Use a reproducible training pipeline (snapshot -> train -> metrics -> artifacts).
- Save a single sklearn Pipeline so backend inference can call predict_proba directly.
- Write metadata.json so backend can show /v1/model/version and /v1/metrics.

Usage example:
python ml/train.py \
  --data ml/data/processed/train.csv \
  --schema backend/app/ml/feature_schema.json \
  --out backend/app/ml/artifacts \
  --version v0.2.0 \
  --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def read_schema(schema_path: Path) -> Tuple[List[str], List[str]]:
    """Return (numeric_features, segment_features) from feature_schema.json."""
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute KS statistic commonly used in credit scoring.
    KS = max_t |CDF_good(t) - CDF_bad(t)| on score distribution.
    """
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]
    # cumulative distributions
    bad = (y_true_sorted == 1).astype(float)
    good = (y_true_sorted == 0).astype(float)

    bad_cum = np.cumsum(bad) / (bad.sum() + 1e-12)
    good_cum = np.cumsum(good) / (good.sum() + 1e-12)

    return float(np.max(np.abs(good_cum - bad_cum)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str, help="Training CSV path (snapshot)")
    parser.add_argument("--schema", required=True, type=str, help="feature_schema.json path")
    parser.add_argument("--out", required=True, type=str, help="Output artifacts directory")
    parser.add_argument("--version", required=True, type=str, help="Model version string, e.g., v0.2.0")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--target", default="default", type=str, help="Target column name")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    schema_path = Path(args.schema).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Available: {list(df.columns)}")

    y = df[args.target].astype(int).to_numpy()

    numeric_features, segment_features = read_schema(schema_path)

    # Keep only columns that exist in df (helps if schema evolves)
    numeric_features = [c for c in numeric_features if c in df.columns]
    segment_features = [c for c in segment_features if c in df.columns]

    if not numeric_features:
        raise ValueError("No numeric features found in data based on schema. Check feature_schema.json.")

    X = df[numeric_features + segment_features].copy()

    # Preprocessing:
    # - numeric: impute median + standardize
    # - categorical (segments): impute most frequent + one-hot
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

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

    # Logistic Regression baseline (classic credit scoring baseline)
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=None,  # lbfgs ignores n_jobs; kept for clarity
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Train/val/test split (stratified)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=args.seed, stratify=y_tmp
    )

    pipeline.fit(X_train, y_train)

    # Predict probabilities for metrics
    val_pd = pipeline.predict_proba(X_val)[:, 1]
    test_pd = pipeline.predict_proba(X_test)[:, 1]

    metrics_val = {
        "roc_auc": float(roc_auc_score(y_val, val_pd)),
        "pr_auc": float(average_precision_score(y_val, val_pd)),
        "ks": float(ks_statistic(y_val, val_pd)),
    }

    metrics_test = {
        "roc_auc": float(roc_auc_score(y_test, test_pd)),
        "pr_auc": float(average_precision_score(y_test, test_pd)),
        "ks": float(ks_statistic(y_test, test_pd)),
    }

    # Save model artifact
    model_path = out_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    # Save metadata (backend reads this for /model/version and /metrics)
    metadata: Dict[str, Any] = {
        "model_name": "logreg_baseline",
        "version": args.version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_schema_sha256": sha256_file(schema_path),
        "features": numeric_features,           # numeric inputs
        "segments": segment_features,           # categorical segment fields used
        "metrics": {
            "val": metrics_val,
            "test": metrics_test,
        },
        "data_info": {
            "n_rows": int(len(df)),
            "default_rate": float(df[args.target].mean()),
            "split": {"train": 0.70, "val": 0.15, "test": 0.15},
            "seed": int(args.seed),
            "data_path": str(data_path),
        },
        # Put default thresholds here early; Phase 3/7 will reuse
        "thresholds": {"approve": 0.30, "reject": 0.40},
        "artifact_files": {"model": "model.joblib", "metadata": "metadata.json"},
    }

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[train] wrote model: {model_path}")
    print(f"[train] wrote metadata: {metadata_path}")
    print(f"[train] val metrics: {metrics_val}")
    print(f"[train] test metrics: {metrics_test}")


if __name__ == "__main__":
    main()
