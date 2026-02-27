# ml/prepare_realdata.py
from __future__ import annotations
#for preparing real raw data
import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ml.mapping.giveme_some_credit import map_gmsc_to_schema
from ml.utils.cleaning import (
    coerce_numeric,
    fill_missing_median,
    missing_summary,
    clip_by_schema,
    winsorize,
)
from ml.utils.metadata import write_json, dataset_summary
from ml.utils.schema import load_schema, feature_names, segment_names, label_name


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw CSV path")
    ap.add_argument("--schema", required=True, help="feature_schema.json path")
    ap.add_argument("--out", required=True, help="output processed snapshot CSV")
    ap.add_argument("--report", default="ml/reports/gmsc_mapping_report.json", help="mapping report json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="gmsc", choices=["gmsc"], help="dataset type")
    ap.add_argument("--do_split", action="store_true")
    ap.add_argument("--split_dir", default="ml/data/processed/split")
    ap.add_argument("--winsorize", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    schema = load_schema(args.schema)
    feats = feature_names(schema)
    segs = segment_names(schema)
    lab = label_name(schema, default="default")

    raw = pd.read_csv(args.raw)

    if args.dataset == "gmsc":
        mapped, notes = map_gmsc_to_schema(raw, schema, rng, label_out=lab)
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    # cleaning only numeric features (segments are categorical)
    mapped = mapped.copy()
    numeric_cols = feats  # features assumed numeric in MVP
    before_missing = missing_summary(mapped, numeric_cols)

    mapped = coerce_numeric(mapped, numeric_cols)
    mapped = fill_missing_median(mapped, numeric_cols)

    if args.winsorize:
        mapped = winsorize(mapped, numeric_cols, 0.01, 0.99)

    mapped = clip_by_schema(mapped, schema, numeric_cols)

    # reorder columns: features + segments + label
    ordered_cols = feats + segs + [lab]
    mapped = mapped[ordered_cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(out_path, index=False)

    # report
    report: Dict[str, Any] = {
        "phase": "2.5",
        "seed": args.seed,
        "raw_path": str(Path(args.raw)),
        "processed_path": str(out_path),
        "schema_path": str(Path(args.schema)),
        "missing_rate_before_fill": before_missing,
        "dataset_summary": dataset_summary(mapped, lab),
        "mapping_notes": notes,
        "cleaning_notes": {
            "fill_strategy": "median",
            "winsorize": bool(args.winsorize),
            "clip_strategy": "schema min/max (if provided)",
        },
    }
    write_json(args.report, report)

    print(f"[OK] processed snapshot: {out_path}")
    print(f"[OK] mapping report: {args.report}")
    print(f"[INFO] rows={len(mapped)} pos_rate={mapped[lab].mean():.4f}")

    if args.do_split:
        from ml.utils.split import stratified_split

        split_dir = Path(args.split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

        train_df, val_df, test_df = stratified_split(mapped, lab, seed=args.seed)
        train_df.to_csv(split_dir / "train.csv", index=False)
        val_df.to_csv(split_dir / "val.csv", index=False)
        test_df.to_csv(split_dir / "test.csv", index=False)

        print(f"[OK] splits written to: {split_dir}")


if __name__ == "__main__":
    main()