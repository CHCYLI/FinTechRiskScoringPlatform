#read feature_schema.json

"""
Generate synthetic consumer-credit style data from feature_schema.json.

Why this exists (Phase 2):
- Phase 2 needs a reproducible data snapshot to train a baseline PD model.
- We generate data aligned with Phase 1 "data contract" (feature_schema.json)
  so the rest of the platform doesn't change when later swap in real datasets.

Output:
- A CSV file with all features + segment fields + target column: `default` (0/1).

Usage example:
python ml/seed_data.py \
  --schema backend/app/ml/feature_schema.json \
  --out ml/data/processed/train.csv \
  --n 50000 --seed 42 --target_default_rate 0.12
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class FeatureDef:
    name: str
    ftype: str
    required: bool
    min_value: float | None
    max_value: float | None


def read_schema(schema_path: Path) -> Tuple[List[FeatureDef], Dict[str, List[Any]]]:
    """Read feature_schema.json and return feature definitions and segments allowed values."""
    raw = json.loads(schema_path.read_text(encoding="utf-8"))

    features: List[FeatureDef] = []
    for f in raw.get("features", []):
        features.append(
            FeatureDef(
                name=f["name"],
                ftype=str(f.get("type", "float")).lower(),
                required=bool(f.get("required", True)),
                min_value=f.get("min", None),
                max_value=f.get("max", None),
            )
        )

    # segments format assumed like:
    # "segments": { "region": ["NE","MW"], "product": ["card","loan"], ... }
    segments: Dict[str, List[Any]] = raw.get("segments", {}) or {}
    return features, segments


def _clip(arr: np.ndarray, min_v: float | None, max_v: float | None) -> np.ndarray:
    if min_v is not None:
        arr = np.maximum(arr, min_v)
    if max_v is not None:
        arr = np.minimum(arr, max_v)
    return arr


def sample_numeric(
    rng: np.random.Generator,
    feature: FeatureDef,
    n: int,
) -> np.ndarray:
    """
    Sample numeric values using heuristics by feature name.
    If unknown, fall back to uniform(min,max) (or standard normal if bounds missing).

    This is intentionally "credit-ish" rather than purely random,
    so the baseline model learns meaningful patterns.
    """
    name = feature.name.lower()
    min_v, max_v = feature.min_value, feature.max_value

    # Helper: produce and clip to schema bounds
    def finish(x: np.ndarray) -> np.ndarray:
        x = _clip(x, min_v, max_v)
        # Cast int-like fields to integers if declared
        if feature.ftype in ("int", "integer"):
            x = np.round(x).astype(int)
        return x

    # Common consumer credit / card features
    if "age" == name:
        x = rng.normal(loc=36, scale=10, size=n)  # typical borrower age center
        return finish(x)

    if "income" in name:
        # lognormal: long right tail; then clip to schema bounds
        x = rng.lognormal(mean=10.7, sigma=0.5, size=n)  # ~ 44k median-ish
        return finish(x)

    if "employment_length" in name or "empl" in name:
        x = rng.integers(low=0, high=21, size=n)  # 0~20 years
        return finish(x.astype(float))

    if "dti" in name:
        # DTI usually 0~0.6-ish; beta gives realistic concentration
        x = rng.beta(a=2.0, b=6.0, size=n) * 0.8
        return finish(x)

    if "utilization" in name or "util" in name:
        x = rng.beta(a=2.0, b=2.5, size=n)  # 0~1
        return finish(x)

    if "delinq" in name:
        x = rng.poisson(lam=0.6, size=n).astype(float)  # mostly 0/1
        return finish(x)

    if "history_length" in name or "credit_history" in name:
        x = rng.gamma(shape=2.0, scale=4.0, size=n)  # years-ish
        return finish(x)

    if "txn_30d" in name or "transactions" in name:
        x = rng.poisson(lam=25, size=n).astype(float)
        return finish(x)

    if "refund_rate" in name or "chargeback" in name:
        x = rng.beta(a=1.2, b=20.0, size=n)  # mostly near 0
        return finish(x)

    if "active_days_30d" in name or "active_days" in name:
        x = rng.integers(low=0, high=31, size=n).astype(float)  # 0~30
        return finish(x)

    # Generic fallback
    if min_v is not None and max_v is not None:
        x = rng.uniform(low=min_v, high=max_v, size=n)
        return finish(x)

    x = rng.normal(loc=0.0, scale=1.0, size=n)
    return finish(x)


def normalize_minmax(x: np.ndarray, min_v: float | None, max_v: float | None) -> np.ndarray:
    """Min-max normalize using schema bounds when available; else standardize roughly."""
    if min_v is not None and max_v is not None and max_v > min_v:
        return (x - min_v) / (max_v - min_v)
    # If no bounds, use robust-ish scaling
    med = np.median(x)
    iqr = np.subtract(*np.percentile(x, [75, 25])) or 1.0
    return (x - med) / iqr


def calibrate_intercept_to_target_mean_pd(
    base_logit: np.ndarray,
    target_default_rate: float,
) -> float:
    """
    Find a constant shift `delta` such that mean(sigmoid(base_logit + delta)) ~= target_default_rate.

    We use binary search; it’s stable and fast for our scale.
    """
    lo, hi = -10.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        mean_pd = float(sigmoid(base_logit + mid).mean())
        if mean_pd < target_default_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_default_label(
    df: pd.DataFrame,
    features: List[FeatureDef],
    target_default_rate: float,
    seed: int,
) -> np.ndarray:
    """
    Create a synthetic default label that behaves like consumer-credit risk:

    Risk increases with:
    - DTI, utilization, delinquencies, refund_rate
    Risk decreases with:
    - income, employment_length, history_length, active_days_30d

    NOTE: This is synthetic; the goal is to create learnable structure, not to mimic any real lender.
    """
    rng = np.random.default_rng(seed)

    # Build weights only for columns that exist in df
    # Using schema bounds for normalization where possible.
    feat_map = {f.name: f for f in features}

    def col(name: str) -> np.ndarray | None:
        return df[name].to_numpy() if name in df.columns else None

    def norm(name: str) -> np.ndarray | None:
        if name not in df.columns:
            return None
        f = feat_map.get(name)
        if not f:
            return normalize_minmax(df[name].to_numpy(), None, None)
        return normalize_minmax(df[name].to_numpy(), f.min_value, f.max_value)

    # Fetch normalized signals if present
    dti = norm("dti")
    util = norm("utilization")
    delinq = norm("delinquencies")
    refund = norm("refund_rate")
    income = norm("income")
    empl = norm("employment_length")
    hist = norm("history_length")
    active = norm("active_days_30d")
    txn = norm("txn_30d")

    # Start with zeros, then add terms that exist
    logit = np.zeros(len(df), dtype=float)

    # Positive risk drivers
    if dti is not None:
        logit += 2.0 * dti
    if util is not None:
        logit += 1.8 * util
    if delinq is not None:
        logit += 2.5 * delinq
    if refund is not None:
        logit += 1.2 * refund

    # Protective drivers
    if income is not None:
        logit += -1.0 * income
    if empl is not None:
        logit += -0.6 * empl
    if hist is not None:
        logit += -0.9 * hist
    if active is not None:
        logit += -0.8 * active
    if txn is not None:
        logit += -0.2 * txn

    # Add noise (unobserved factors)
    logit += rng.normal(0.0, 0.35, size=len(df))

    # Calibrate intercept to get the target average default rate
    delta = calibrate_intercept_to_target_mean_pd(logit, target_default_rate)
    pd_hat = sigmoid(logit + delta)

    y = rng.binomial(n=1, p=pd_hat, size=len(df))
    return y.astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True, type=str, help="Path to backend/app/ml/feature_schema.json")
    parser.add_argument("--out", required=True, type=str, help="Output CSV path, e.g., ml/data/processed/train.csv")
    parser.add_argument("--n", default=50000, type=int, help="Number of rows")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--target_default_rate", default=0.12, type=float, help="Average default rate to simulate (0~1)")
    args = parser.parse_args()

    schema_path = Path(args.schema).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    features, segments = read_schema(schema_path)
    rng = np.random.default_rng(args.seed)

    data: Dict[str, Any] = {}

    # 1) Sample numeric features from schema definitions
    for f in features:
        # In Phase 2 we generate everything, even if optional; optional fields add realism
        if f.ftype in ("int", "integer", "float", "number", "double"):
            data[f.name] = sample_numeric(rng, f, args.n)
        else:
            # If schema contains non-numeric types here, we skip (segments handled separately)
            pass

    # 2) Sample segment categorical fields (channel/region/product etc.)
    # We sample from allowed values; if empty, we skip.
    for seg_name, allowed in segments.items():
        if not allowed:
            continue
        # Weighted sampling can be added later; uniform for MVP
        data[seg_name] = rng.choice(allowed, size=args.n, replace=True)

    df = pd.DataFrame(data)

    # 3) Build synthetic label
    df["default"] = build_default_label(df, features, args.target_default_rate, seed=args.seed)

    # 4) Save snapshot
    df.to_csv(out_path, index=False)

    print(f"[seed_data] wrote: {out_path}")
    print(f"[seed_data] rows: {len(df)}  default_rate: {df['default'].mean():.4f}")
    print(f"[seed_data] columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
