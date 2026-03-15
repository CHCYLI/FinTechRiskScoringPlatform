from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

STATUS_COL = "loan_status"


def _parse_issue_year(issue_month: pd.Series) -> pd.Series:
    s = issue_month.astype("string")
    dt = pd.to_datetime(s, errors="coerce", format="%Y-%m")  # try YYYY-MM first
    dt2 = pd.to_datetime(s, errors="coerce")                 # fallback
    dt = dt.fillna(dt2)
    return dt.dt.year


def _safe_divide(num: pd.Series, den: pd.Series, default: float = 0.0) -> pd.Series:
    den = den.replace(0, np.nan)
    out = (num / den).replace([np.inf, -np.inf], np.nan).fillna(default)
    return out


def _stable_encode(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    s = series.astype("string")
    vals = sorted([v for v in s.dropna().unique().tolist() if v != "<NA>"])
    mapping = {v: i + 1 for i, v in enumerate(vals)}
    codes = s.map(mapping).fillna(0).astype(int)
    return codes, mapping


def _schema_numeric_features(schema: Optional[dict]) -> List[str]:
    """
    Match your train.py schema parser:
    raw["features"] is a list of dicts: {"name": "...", "type": "..."}.
    """
    if not schema:
        return []
    feats = schema.get("features", [])
    out: List[str] = []
    if isinstance(feats, list):
        for f in feats:
            if not isinstance(f, dict) or "name" not in f:
                continue
            ftype = str(f.get("type", "float")).lower()
            if ftype in ("int", "integer", "float", "number", "double"):
                out.append(str(f["name"]))
    return out


def _schema_segments(schema: Optional[dict]) -> List[str]:
    if not schema:
        return ["channel", "region", "product"]
    segs = schema.get("segments", {}) or {}
    if isinstance(segs, dict):
        return list(segs.keys())
    return ["channel", "region", "product"]


def _schema_label(schema: Optional[dict]) -> str:
    if not schema:
        return "default"
    lab = schema.get("label")
    if isinstance(lab, dict) and "name" in lab:
        return str(lab["name"])
    if isinstance(lab, str):
        return lab
    return "default"


def _allowed(schema: Optional[dict], seg: str) -> Optional[List[str]]:
    if not schema:
        return None
    cfg = schema.get("segments", {}).get(seg, {})
    if not isinstance(cfg, dict):
        return None
    vals = cfg.get("allowed") or cfg.get("values")
    if vals is None:
        return None
    return [str(x) for x in vals]


def _make_label(df: pd.DataFrame, label_mode: str) -> Tuple[pd.Series, pd.Series, dict]:
    if STATUS_COL not in df.columns:
        raise ValueError(f"Missing required column: {STATUS_COL}")

    st = df[STATUS_COL].astype("string")
    good = {"Fully Paid"}
    bad = {"Charged Off", "Default"}

    is_late = st.fillna("").str.contains("Late", case=False, regex=False)
    is_bad = st.isin(list(bad)) | is_late
    is_good = st.isin(list(good))

    if label_mode == "include_current_as_good":
        is_good = is_good | (st == "Current")

    keep = is_good | is_bad
    y = pd.Series(np.where(is_bad, 1, 0), index=df.index).astype(int)

    notes = {
        "label_mode": label_mode,
        "good_statuses": sorted(list(good)) + (["Current"] if label_mode == "include_current_as_good" else []),
        "bad_statuses": sorted(list(bad)) + ["*Late*"],
        "dropped_statuses_sample": sorted(list(st[~keep].dropna().unique()[:10])),
    }
    return y, keep, notes


def _infer_feature(df: pd.DataFrame, feat: str, computed: Dict[str, pd.Series]) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Map schema feature name -> series from LendingClub loans_full_schema columns.
    If not found, return reasonable default and annotate in note.
    """
    f = feat.lower()

    # direct match first
    if feat in df.columns:
        s = df[feat]
        if s.dtype == "object" or str(s.dtype).startswith("string"):
            codes, mapping = _stable_encode(s)
            return codes, {"kind": "categorical_encoded", "source": feat, "mapping_size": len(mapping)}
        return s, {"kind": "direct", "source": feat}

    # income
    if "income" in f:
        if "annual_income_joint" in df.columns and df["annual_income_joint"].notna().any():
            return df["annual_income_joint"].fillna(df.get("annual_income")), {"kind": "mapped", "source": "annual_income_joint->annual_income"}
        if "annual_income" in df.columns:
            return df["annual_income"], {"kind": "mapped", "source": "annual_income"}

    # employment length
    if ("emp" in f and "length" in f) or ("employment" in f and "length" in f):
        if "emp_length" in df.columns:
            return df["emp_length"], {"kind": "mapped", "source": "emp_length"}

    # dti
    if "dti" in f or "debt_to_income" in f:
        if "debt_to_income_joint" in df.columns and df["debt_to_income_joint"].notna().any():
            return df["debt_to_income_joint"].fillna(df.get("debt_to_income")), {"kind": "mapped", "source": "debt_to_income_joint->debt_to_income"}
        if "debt_to_income" in df.columns:
            return df["debt_to_income"], {"kind": "mapped", "source": "debt_to_income"}

    # utilization
    if "util" in f:
        if "utilization" in computed:
            return computed["utilization"], {"kind": "computed", "source": "total_credit_utilized/total_credit_limit"}
        if "total_credit_utilized" in df.columns and "total_credit_limit" in df.columns:
            s = _safe_divide(df["total_credit_utilized"], df["total_credit_limit"], default=0.0)
            return s, {"kind": "computed", "source": "total_credit_utilized/total_credit_limit"}

    # delinquencies
    if "delinq" in f or "delin" in f:
        if "delinq_2y" in df.columns:
            return df["delinq_2y"], {"kind": "mapped", "source": "delinq_2y"}
        if "current_accounts_delinq" in df.columns:
            return df["current_accounts_delinq"], {"kind": "proxy", "source": "current_accounts_delinq"}

    # credit history length
    if "history" in f and ("length" in f or "year" in f or "years" in f):
        if "history_length_years" in computed:
            return computed["history_length_years"], {"kind": "computed", "source": "issue_year - earliest_credit_line"}
        return pd.Series([5] * len(df), index=df.index), {"kind": "default", "value": 5, "reason": "no history fields"}

    # inquiries
    if "inquir" in f and "inquiries_last_12m" in df.columns:
        return df["inquiries_last_12m"], {"kind": "mapped", "source": "inquiries_last_12m"}

    # open/total lines
    if "open_credit_lines" in f and "open_credit_lines" in df.columns:
        return df["open_credit_lines"], {"kind": "direct", "source": "open_credit_lines"}
    if "total_credit_lines" in f and "total_credit_lines" in df.columns:
        return df["total_credit_lines"], {"kind": "direct", "source": "total_credit_lines"}

    # loan attributes
    if ("loan" in f and "amount" in f) and "loan_amount" in df.columns:
        return df["loan_amount"], {"kind": "direct", "source": "loan_amount"}
    if "interest" in f and "rate" in f and "interest_rate" in df.columns:
        return df["interest_rate"], {"kind": "direct", "source": "interest_rate"}
    if "installment" in f and "installment" in df.columns:
        return df["installment"], {"kind": "direct", "source": "installment"}
    if "term" in f and "term" in df.columns:
        s = df["term"]
        if s.dtype == "object" or str(s.dtype).startswith("string"):
            num = pd.to_numeric(s.astype("string").str.extract(r"(\d+)")[0], errors="coerce")
            return num, {"kind": "mapped", "source": "term (extract digits)"}
        return s, {"kind": "direct", "source": "term"}

    # behavior-like fields not present -> defaults
    if "tx" in f or "transaction" in f:
        return pd.Series([10] * len(df), index=df.index), {"kind": "default", "value": 10}
    if "refund" in f:
        return pd.Series([0.0] * len(df), index=df.index), {"kind": "default", "value": 0.0}
    if "active" in f and "day" in f:
        return pd.Series([20] * len(df), index=df.index), {"kind": "default", "value": 20}

    # last resort
    return pd.Series([0.0] * len(df), index=df.index), {"kind": "default", "value": 0.0, "reason": "no matching raw column"}


def map_lendingclub_loans_full_schema(
    raw: pd.DataFrame,
    schema: Optional[dict],
    seed: int = 42,
    label_mode: str = "chargedoff_vs_fullypaid",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    numeric_feats = _schema_numeric_features(schema)
    segs = _schema_segments(schema)
    label_col = _schema_label(schema)

    if not numeric_feats:
        raise ValueError(
            "Your schema has no numeric features in raw['features'] list. "
            "Check feature_schema.json format: features must be list of {name,type}."
        )

    y_all, keep, label_notes = _make_label(raw, label_mode)
    df = raw.loc[keep].copy()
    y = y_all.loc[keep]

    # computed helpers
    computed: Dict[str, pd.Series] = {}
    if "total_credit_utilized" in df.columns and "total_credit_limit" in df.columns:
        computed["utilization"] = _safe_divide(df["total_credit_utilized"], df["total_credit_limit"], default=0.0)

    if "earliest_credit_line" in df.columns:
        earliest_year = pd.to_numeric(df["earliest_credit_line"], errors="coerce")
        if "issue_month" in df.columns:
            issue_year = _parse_issue_year(df["issue_month"])
        else:
            issue_year = pd.Series([2018] * len(df), index=df.index)
        fallback_year = int(issue_year.dropna().max()) if issue_year.notna().any() else 2018
        issue_year = issue_year.fillna(fallback_year)
        computed["history_length_years"] = (issue_year - earliest_year).clip(lower=0).fillna(0)

    out = pd.DataFrame(index=df.index)

    notes: Dict[str, Any] = {
        "dataset": "loans_full_schema (LendingClub)",
        "label_notes": label_notes,
        "feature_notes": {},
        "segment_notes": {},
        "used_numeric_features": numeric_feats,
    }

    # build numeric feature columns exactly matching schema names
    for feat in numeric_feats:
        s, note = _infer_feature(df, feat, computed)
        out[feat] = s
        notes["feature_notes"][feat] = note

    # segments (categorical)
    for seg in segs:
        allowed = _allowed(schema, seg)
        sl = seg.lower()

        if sl in {"region", "state"} and "state" in df.columns:
            s = df["state"].astype("string").fillna("unknown")
            if allowed:
                s = s.where(s.isin(allowed), other=rng.choice(allowed, size=len(s)))
            out[seg] = s
            notes["segment_notes"][seg] = {"source": "state", "allowed_used": bool(allowed)}
            continue

        if sl == "product" and "loan_purpose" in df.columns:
            s = df["loan_purpose"].astype("string").fillna("unknown")
            if allowed:
                s = s.where(s.isin(allowed), other=rng.choice(allowed, size=len(s)))
            out[seg] = s
            notes["segment_notes"][seg] = {"source": "loan_purpose", "allowed_used": bool(allowed)}
            continue

        if sl == "channel":
            src = None
            for cand in ["initial_listing_status", "disbursement_method", "application_type"]:
                if cand in df.columns:
                    src = cand
                    break
            if src:
                s = df[src].astype("string").fillna("unknown")
                if allowed:
                    s = s.where(s.isin(allowed), other=rng.choice(allowed, size=len(s)))
                out[seg] = s
                notes["segment_notes"][seg] = {"source": src, "allowed_used": bool(allowed)}
            else:
                vals = allowed or ["online", "branch"]
                out[seg] = pd.Series(rng.choice(vals, size=len(df)), index=df.index)
                notes["segment_notes"][seg] = {"source": "synthetic", "values": vals}
            continue

        # fallback synthetic
        vals = allowed or ["unknown"]
        out[seg] = pd.Series(rng.choice(vals, size=len(df)), index=df.index)
        notes["segment_notes"][seg] = {"source": "synthetic_fallback", "values": vals}

    out[label_col] = y.values
    return out, notes
