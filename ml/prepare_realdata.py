# ml/prepare_realdata.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.mapping.giveme_some_credit import map_gmsc_to_schema
from ml.mapping.lendingclub_loans_full_schema import map_lc_loans_to_schema
from ml.utils.cleaning import (
    coerce_numeric,
    fill_missing_median,
    missing_summary,
    clip_by_schema,
    winsorize,
)
from ml.utils.metadata import write_json, dataset_summary
from ml.utils.schema import load_schema, feature_names, segment_names, label_name


def parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw CSV path")
    ap.add_argument("--schema", required=True, help="feature_schema.json path")
    ap.add_argument("--out", required=True, help="output processed snapshot CSV")
    ap.add_argument("--report", default="ml/reports/mapping_report.json", help="mapping report json")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dataset", default="lc_loans", choices=["gmsc", "lc_loans"])
    # label controls (for LendingClub)
    ap.add_argument("--label_mode", default="chargedoff_vs_fullypaid",
                    choices=["chargedoff_vs_fullypaid", "include_current_as_good", "custom"])
    ap.add_argument("--good_statuses", default=None, help="comma-separated, label_mode=custom only")
    ap.add_argument("--bad_statuses", default=None, help="comma-separated, label_mode=custom only")

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

    elif args.dataset == "lc_loans":
        good = parse_csv_list(args.good_statuses)
        bad = parse_csv_list(args.bad_statuses)
        mapped, notes = map_lc_loans_to_schema(
            raw,
            schema,
            rng,
            label_out=lab,
            label_mode=args.label_mode,
            good_statuses=good,
            bad_statuses=bad,
        )
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    # cleaning only numeric features (segments are categorical)
    numeric_cols = feats  # features assumed numeric after mapping (categoricals encoded)
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

    report: Dict[str, Any] = {
        "phase": "2.5",
        "dataset": args.dataset,
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