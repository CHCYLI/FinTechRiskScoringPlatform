from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

# allow running from repo root: python ml/prepare_realdata.py ...
try:
    from ml.mapping.lendingclub_loans_full_schema import map_lendingclub_loans_full_schema
except Exception:
    from mapping.lendingclub_loans_full_schema import map_lendingclub_loans_full_schema


def load_schema(schema_path: Optional[str]) -> Optional[dict]:
    if not schema_path:
        return None
    p = Path(schema_path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fill_missing_median(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        med = out[c].median()
        if pd.isna(med):
            med = 0.0
        out[c] = out[c].fillna(med)
    return out


def schema_numeric_features(schema: Optional[dict]) -> list[str]:
    if not schema:
        return []
    feats = schema.get("features", [])
    out = []
    if isinstance(feats, list):
        for f in feats:
            if not isinstance(f, dict) or "name" not in f:
                continue
            ftype = str(f.get("type", "float")).lower()
            if ftype in ("int", "integer", "float", "number", "double"):
                out.append(str(f["name"]))
    return out


def schema_segments(schema: Optional[dict]) -> list[str]:
    if not schema:
        return ["channel", "region", "product"]
    segs = schema.get("segments", {}) or {}
    if isinstance(segs, dict):
        return list(segs.keys())
    return ["channel", "region", "product"]


def schema_label(schema: Optional[dict]) -> str:
    if not schema:
        return "default"
    lab = schema.get("label")
    if isinstance(lab, dict) and "name" in lab:
        return str(lab["name"])
    if isinstance(lab, str):
        return lab
    return "default"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--schema", default="backend/app/ml/feature_schema.json")
    ap.add_argument("--out", default="ml/data/processed/train_real.csv")
    ap.add_argument("--report", default="ml/data/processed/lc_mapping_report.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_mode", default="chargedoff_vs_fullypaid",
                    choices=["chargedoff_vs_fullypaid", "include_current_as_good"])
    return ap.parse_args()


def main():
    args = parse_args()
    schema = load_schema(args.schema)
    raw = pd.read_csv(args.raw)

    mapped, notes = map_lendingclub_loans_full_schema(
        raw=raw,
        schema=schema,
        seed=args.seed,
        label_mode=args.label_mode,
    )

    # clean numeric feature cols according to schema
    num_cols = schema_numeric_features(schema)
    if not num_cols:
        raise ValueError("No numeric features found in schema.features list. Fix feature_schema.json first.")

    mapped = coerce_numeric(mapped, num_cols)
    mapped = fill_missing_median(mapped, num_cols)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(out_path, index=False)

    label_col = schema_label(schema)
    seg_cols = schema_segments(schema)

    report = {
        "phase": "2.5",
        "dataset": "loans_full_schema (LendingClub)",
        "raw_path": str(Path(args.raw)),
        "schema_path": args.schema,
        "processed_path": str(out_path),
        "seed": args.seed,
        "label_mode": args.label_mode,
        "rows": int(len(mapped)),
        "pos_rate": float(mapped[label_col].mean()),
        "numeric_features": num_cols,
        "segments": [c for c in seg_cols if c in mapped.columns],
        "columns": list(mapped.columns),
        "mapping_notes": notes,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] wrote: {out_path}")
    print(f"[OK] report: {report_path}")
    print(f"[INFO] rows={len(mapped)} cols={len(mapped.columns)} numeric_features={len(num_cols)}")


if __name__ == "__main__":
    main()
