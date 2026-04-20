from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.config import PROJECT_ROOT, get_settings
from app.services.inference import _extract_thresholds, decide, get_model_bundle


def _load_portfolio_df() -> pd.DataFrame:
    settings = get_settings()
    data_path = Path(settings.portfolio_data_path)
    if not data_path.exists():
        legacy_default = (PROJECT_ROOT / "ml" / "data" / "processed" / "train_real.csv").resolve()
        if data_path.name == "portfolio_sample.csv" and legacy_default.exists():
            data_path = legacy_default
    if not data_path.exists():
        raise FileNotFoundError(f"Portfolio data not found: {data_path}")
    return pd.read_csv(data_path)


def build_portfolio_summary(
    group_by: str,
    region: Optional[str] = None,
    channel: Optional[str] = None,
    product: Optional[str] = None,
    limit: int = 50,
) -> dict:
    bundle = get_model_bundle()
    metadata = bundle.metadata
    features = bundle.features
    segments = bundle.segments

    if group_by not in segments:
        raise ValueError(f"group_by must be one of {segments}")

    df = _load_portfolio_df().copy()

    # Ensure potential group/filter fields exist to avoid KeyError.
    for segment in segments:
        if segment not in df.columns:
            df[segment] = None

    if region is not None:
        df = df[df["region"] == region]
    if channel is not None:
        df = df[df["channel"] == channel]
    if product is not None:
        df = df[df["product"] == product]

    if df.empty:
        return {
            "model_version": str(metadata.get("version", "unknown")),
            "group_by": group_by,
            "filters": {
                "region": region,
                "channel": channel,
                "product": product,
            },
            "rows": [],
        }

    required_cols = features + segments
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    X = df[required_cols]
    pd_values = bundle.model.predict_proba(X)[:, 1]

    approve_th, reject_th, _ = _extract_thresholds(metadata)

    df["pd"] = pd_values
    df["decision"] = df["pd"].apply(lambda x: decide(float(x), approve_th, reject_th))
    df["approve_flag"] = (df["decision"] == "Approve").astype(int)
    df["review_flag"] = (df["decision"] == "Review").astype(int)
    df["reject_flag"] = (df["decision"] == "Reject").astype(int)

    agg_map = {
        "count": ("pd", "size"),
        "avg_pd": ("pd", "mean"),
        "approve_count": ("approve_flag", "sum"),
        "review_count": ("review_flag", "sum"),
        "reject_count": ("reject_flag", "sum"),
    }
    if "default" in df.columns:
        agg_map["bad_rate"] = ("default", "mean")

    grouped = df.groupby(group_by, dropna=False).agg(**agg_map).reset_index()
    grouped = grouped.rename(columns={group_by: "group"})

    if "bad_rate" not in grouped.columns:
        grouped["bad_rate"] = None

    grouped["approve_rate"] = grouped["approve_count"] / grouped["count"]
    grouped["review_rate"] = grouped["review_count"] / grouped["count"]
    grouped["reject_rate"] = grouped["reject_count"] / grouped["count"]

    grouped = grouped.sort_values(["avg_pd", "count"], ascending=[False, False]).head(limit)

    rows = []
    for _, row in grouped.iterrows():
        bad_rate = row["bad_rate"]
        rows.append(
            {
                "group": "UNKNOWN" if pd.isna(row["group"]) else str(row["group"]),
                "count": int(row["count"]),
                "avg_pd": float(row["avg_pd"]),
                "approve_count": int(row["approve_count"]),
                "review_count": int(row["review_count"]),
                "reject_count": int(row["reject_count"]),
                "approve_rate": float(row["approve_rate"]),
                "review_rate": float(row["review_rate"]),
                "reject_rate": float(row["reject_rate"]),
                "bad_rate": float(bad_rate) if pd.notna(bad_rate) else None,
            }
        )

    return {
        "model_version": str(metadata.get("version", "unknown")),
        "group_by": group_by,
        "filters": {
            "region": region,
            "channel": channel,
            "product": product,
        },
        "rows": rows,
    }
