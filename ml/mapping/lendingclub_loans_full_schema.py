# ml/mapping/lendingclub_loans_full_schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.utils.schema import (
    feature_names,
    segment_names,
    allowed_values_for_segment,
)

STATUS_COL = "loan_status"

DEFAULT_GOOD = {"Fully Paid"}
DEFAULT_BAD = {"Charged Off", "Default"}


def _parse_issue_year(issue_month: pd.Series) -> pd.Series:
    """
    issue_month examples might be like:
      - '2018-03'
      - 'Mar-2018'
      - '2018 Mar'
    We'll try pandas to_datetime.
    """
    dt = pd.to_datetime(issue_month, errors="coerce", infer_datetime_format=True)
    year = dt.dt.year
    return year


def _safe_divide(num: pd.Series, den: pd.Series, default: float = 0.0) -> pd.Series:
    den = den.replace(0, np.nan)
    out = num / den
    out = out.replace([np.inf, -np.inf], np.nan).fillna(default)
    return out


def _stable_encode(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Stable label encoding for categorical feature -> int codes.
    Unknown/NaN -> 0; others -> 1..K (sorted by string)
    """
    s = series.astype("string")
    vals = sorted([v for v in s.dropna().unique().tolist() if v != "<NA>"])
    mapping = {v: i + 1 for i, v in enumerate(vals)}
    codes = s.map(mapping).fillna(0).astype(int)
    return codes, mapping


def _make_label(
    raw: pd.DataFrame,
    label_mode: str,
    good_statuses: Optional[List[str]],
    bad_statuses: Optional[List[str]],
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """
    Returns (label, keep_mask, notes)
    label_mode:
      - "chargedoff_vs_fullypaid" (default): keep only Fully Paid vs (Charged Off/Default/Late*)
      - "include_current_as_good": keep Current as good too
      - "custom": use provided good_statuses/bad_statuses; drop others
    """
    if STATUS_COL not in raw.columns:
        raise ValueError(f"Missing required column: {STATUS_COL}")

    st = raw[STATUS_COL].astype("string")

    notes: Dict[str, Any] = {"label_mode": label_mode}

    # Build sets
    if label_mode == "custom":
        if not good_statuses or not bad_statuses:
            raise ValueError("custom label_mode requires --good_statuses and --bad_statuses")
        good = set(good_statuses)
        bad = set(bad_statuses)
    else:
        good = set(DEFAULT_GOOD)
        bad = set(DEFAULT_BAD)
        # Anything that contains 'Late' counts as bad by default
        # (covers 'Late (16-30 days)' etc.)
        notes["auto_bad_contains"] = ["Late"]

    is_late = st.fillna("").str.contains("Late", case=False, regex=False)
    is_bad = st.isin(list(bad)) | is_late
    is_good = st.isin(list(good))

    if label_mode == "include_current_as_good":
        is_good = is_good | (st == "Current")
        notes["included_as_good"] = ["Current"]

    keep = is_good | is_bad
    label = pd.Series(np.where(is_bad, 1, 0), index=raw.index).astype(int)

    notes["good_statuses"] = sorted(list(good))
    notes["bad_statuses"] = sorted(list(bad))
    notes["dropped_statuses_sample"] = sorted(list(st[~keep].dropna().unique()[:10]))

    return label, keep, notes


def _infer_feature(
    raw: pd.DataFrame,
    feat: str,
    computed: Dict[str, pd.Series],
) -> Tuple[pd.Series, Optional[Dict[str, Any]]]:
    """
    Returns (series, note) where note records proxy/default/encoding usage.
    """
    f = feat.lower()

    # Direct column name match
    if feat in raw.columns:
        s = raw[feat]
        # if categorical -> encode
        if s.dtype == "object" or str(s.dtype).startswith("string"):
            codes, mapping = _stable_encode(s)
            return codes, {"type": "categorical_encoded", "mapping": mapping}
        return s, None

    # Income
    if "income" in f:
        if "annual_income_joint" in raw.columns and raw["annual_income_joint"].notna().any():
            s = raw["annual_income_joint"].fillna(raw.get("annual_income"))
            return s, {"type": "mapped", "source": "annual_income_joint (fallback annual_income)"}
        if "annual_income" in raw.columns:
            return raw["annual_income"], {"type": "mapped", "source": "annual_income"}

    # Employment length
    if ("emp" in f and "length" in f) or ("employment" in f and "length" in f):
        if "emp_length" in raw.columns:
            return raw["emp_length"], {"type": "mapped", "source": "emp_length"}

    # DTI
    if "dti" in f or "debt_to_income" in f:
        if "debt_to_income_joint" in raw.columns and raw["debt_to_income_joint"].notna().any():
            s = raw["debt_to_income_joint"].fillna(raw.get("debt_to_income"))
            return s, {"type": "mapped", "source": "debt_to_income_joint (fallback debt_to_income)"}
        if "debt_to_income" in raw.columns:
            return raw["debt_to_income"], {"type": "mapped", "source": "debt_to_income"}

    # Utilization
    if "util" in f:
        if "utilization" in computed:
            return computed["utilization"], {"type": "computed", "source": "total_credit_utilized/total_credit_limit"}
        # fallback: if schema expects utilized or limit separately
        if "total_credit_utilized" in raw.columns and "total_credit_limit" in raw.columns:
            s = _safe_divide(raw["total_credit_utilized"], raw["total_credit_limit"], default=0.0)
            return s, {"type": "computed", "source": "total_credit_utilized/total_credit_limit"}

    # Delinquencies
    if "delinq" in f or "delin" in f:
        if "delinq_2y" in raw.columns:
            return raw["delinq_2y"], {"type": "mapped", "source": "delinq_2y"}
        if "current_accounts_delinq" in raw.columns:
            return raw["current_accounts_delinq"], {"type": "proxy", "source": "current_accounts_delinq"}

    # Credit history length
    if "history" in f and ("length" in f or "years" in f or "months" in f):
        if "credit_history_years" in computed:
            return computed["credit_history_years"], {"type": "computed", "source": "issue_year - earliest_credit_line"}
        # fallback default
        return pd.Series([60] * len(raw), index=raw.index), {"type": "default", "value": 60, "reason": "no credit history fields"}

    # Inquiries
    if "inquir" in f:
        if "inquiries_last_12m" in raw.columns:
            return raw["inquiries_last_12m"], {"type": "mapped", "source": "inquiries_last_12m"}

    # Loan amount / term / interest
    if "loan_amount" in f or (("amount" in f) and ("loan" in f)):
        if "loan_amount" in raw.columns:
            return raw["loan_amount"], {"type": "mapped", "source": "loan_amount"}
    if "interest" in f and "rate" in f:
        if "interest_rate" in raw.columns:
            return raw["interest_rate"], {"type": "mapped", "source": "interest_rate"}
    if "installment" in f:
        if "installment" in raw.columns:
            return raw["installment"], {"type": "mapped", "source": "installment"}
    if "term" in f:
        if "term" in raw.columns:
            # if term is like "36 months", encode numeric
            s = raw["term"]
            if s.dtype == "object" or str(s.dtype).startswith("string"):
                num = pd.to_numeric(s.astype("string").str.extract(r"(\d+)")[0], errors="coerce")
                return num, {"type": "mapped", "source": "term (extracted months)"}
            return s, {"type": "mapped", "source": "term"}

    # Behavioral features not in dataset -> defaults
    if "tx" in f or "transaction" in f:
        return pd.Series([10] * len(raw), index=raw.index), {"type": "default", "value": 10, "reason": "no tx behavior in dataset"}
    if "refund" in f:
        return pd.Series([0.0] * len(raw), index=raw.index), {"type": "default", "value": 0.0, "reason": "no refund behavior in dataset"}
    if "active" in f and "day" in f:
        return pd.Series([20] * len(raw), index=raw.index), {"type": "default", "value": 20, "reason": "no activity days in dataset"}

    # Fallback default
    return pd.Series([0.0] * len(raw), index=raw.index), {"type": "default", "value": 0.0, "reason": "no matching raw column"}


def map_lc_loans_to_schema(
    raw: pd.DataFrame,
    schema: Dict[str, Any],
    rng: np.random.Generator,
    label_out: str = "default",
    label_mode: str = "chargedoff_vs_fullypaid",
    good_statuses: Optional[List[str]] = None,
    bad_statuses: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    feats = feature_names(schema)
    segs = segment_names(schema)

    # label + keep mask
    label, keep, label_notes = _make_label(raw, label_mode, good_statuses, bad_statuses)
    df = raw.loc[keep].copy()
    y = label.loc[keep]

    # computed helpers
    computed: Dict[str, pd.Series] = {}

    if "total_credit_utilized" in df.columns and "total_credit_limit" in df.columns:
        computed["utilization"] = _safe_divide(df["total_credit_utilized"], df["total_credit_limit"], default=0.0)

    if "earliest_credit_line" in df.columns:
        # earliest_credit_line is year per your description
        earliest_year = pd.to_numeric(df["earliest_credit_line"], errors="coerce")
        issue_year = None
        if "issue_month" in df.columns:
            issue_year = _parse_issue_year(df["issue_month"])
        # fallback issue year: use max year in issue_year if exists else 2018
        fallback_year = int(issue_year.dropna().max()) if issue_year is not None and issue_year.notna().any() else 2018
        if issue_year is None:
            issue_year = pd.Series([fallback_year] * len(df), index=df.index)
        issue_year = issue_year.fillna(fallback_year)
        computed["credit_history_years"] = (issue_year - earliest_year).clip(lower=0).fillna(0)

    out = pd.DataFrame(index=df.index)
    notes: Dict[str, Any] = {
        "dataset_name": "LendingClub loans_full_schema",
        "label": label_notes,
        "feature_notes": {},
        "segments_notes": {},
    }

    # features
    for feat in feats:
        s, note = _infer_feature(df, feat, computed)
        out[feat] = s
        if note:
            notes["feature_notes"][feat] = note

    # segments
    for seg in segs:
        allowed = allowed_values_for_segment(schema, seg)
        seg_lower = seg.lower()

        if seg_lower in {"region", "state"} and "state" in df.columns:
            s = df["state"].astype("string").fillna("unknown")
            if allowed:
                # if allowed contains state codes, keep those; else fallback to random allowed
                if all(len(v) == 2 for v in allowed):
                    s = s.where(s.isin(allowed), other=rng.choice(allowed, size=len(s)))
                else:
                    s = pd.Series(rng.choice(allowed, size=len(s)), index=df.index)
            out[seg] = s
            notes["segments_notes"][seg] = {"source": "state", "allowed_used": bool(allowed)}
            continue

        if seg_lower in {"product"} and "loan_purpose" in df.columns:
            s = df["loan_purpose"].astype("string").fillna("unknown")
            if allowed:
                s = s.where(s.isin(allowed), other=rng.choice(allowed, size=len(s)))
            out[seg] = s
            notes["segments_notes"][seg] = {"source": "loan_purpose", "allowed_used": bool(allowed)}
            continue

        if seg_lower in {"channel"}:
            src_col = None
            for cand in ["initial_listing_status", "disbursement_method", "application_type"]:
                if cand in df.columns:
                    src_col = cand
                    break
            if src_col:
                s = df[src_col].astype("string").fillna("unknown")
                if allowed:
                    s = s.where(s.isin(allowed), other=rng.choice(allowed, size=len(s)))
                out[seg] = s
                notes["segments_notes"][seg] = {"source": src_col, "allowed_used": bool(allowed)}
            else:
                vals = allowed or ["online", "branch"]
                out[seg] = pd.Series(rng.choice(vals, size=len(df)), index=df.index)
                notes["segments_notes"][seg] = {"source": "synthetic", "values": vals}
            continue

        # amount band / bucket
        if ("amount" in seg_lower) and ("band" in seg_lower or "bucket" in seg_lower) and "loan_amount" in df.columns:
            amt = pd.to_numeric(df["loan_amount"], errors="coerce").fillna(0)
            bins = [0, 5000, 10000, 20000, 99999999]
            labels = ["0-5k", "5-10k", "10-20k", "20k+"]
            band = pd.cut(amt, bins=bins, labels=labels, include_lowest=True).astype("string")
            if allowed:
                band = band.where(band.isin(allowed), other=rng.choice(allowed, size=len(band)))
            out[seg] = band
            notes["segments_notes"][seg] = {"source": "loan_amount->band", "allowed_used": bool(allowed)}
            continue

        # fallback: synthesize
        vals = allowed or ["unknown"]
        out[seg] = pd.Series(rng.choice(vals, size=len(df)), index=df.index)
        notes["segments_notes"][seg] = {"source": "synthetic_fallback", "values": vals}

    out[label_out] = y.values
    return out, notes