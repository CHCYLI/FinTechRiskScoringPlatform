# backend/app/services/inference.py
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "ml" / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACT_DIR / "model.joblib"
DEFAULT_META_PATH = ARTIFACT_DIR / "metadata.json"


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    metadata: Dict[str, Any]
    features: List[str]
    segments: List[str]


def _load_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_columns(metadata: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Support different metadata shapes gracefully.
    Expected from your summary:
      metadata["features"], metadata["segments"]
    """
    features = metadata.get("features") or []
    segments = metadata.get("segments") or []
    if not isinstance(features, list):
        features = []
    if not isinstance(segments, list):
        segments = []
    return features, segments


def _extract_thresholds(metadata: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
    """
    Prefer metadata thresholds; fallback to defaults if missing.
    We support a few possible key names to avoid breaking.
    """
    t = metadata.get("thresholds") or {}
    approve = None
    reject = None

    # common patterns
    if isinstance(t, dict):
        approve = t.get("approve") or t.get("approve_threshold") or t.get("approve_pd")
        reject = t.get("reject") or t.get("reject_threshold") or t.get("reject_pd")

        # sometimes stored as {"approve": {"pd": 0.3}, ...}
        if isinstance(approve, dict):
            approve = approve.get("pd")
        if isinstance(reject, dict):
            reject = reject.get("pd")

    # fallback defaults
    approve = float(approve) if approve is not None else 0.30
    reject = float(reject) if reject is not None else 0.40

    # normalize back
    thresholds = {"approve": approve, "reject": reject}
    return approve, reject, thresholds


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {DEFAULT_MODEL_PATH}")

    model = joblib.load(DEFAULT_MODEL_PATH)
    metadata = _load_metadata(DEFAULT_META_PATH)

    features, segments = _extract_columns(metadata)
    if not features:
        # last resort: infer from schema hash / pipeline is hard, so force explicit
        raise ValueError("metadata.features is missing or empty. Please ensure metadata.json contains 'features' list.")
    # segments can be empty in theory, but your contract includes them
    return ModelBundle(model=model, metadata=metadata, features=features, segments=segments)


def applicant_to_df(applicant: Dict[str, Any], bundle: ModelBundle) -> pd.DataFrame:
    cols = bundle.features + bundle.segments
    row = {c: applicant.get(c, None) for c in cols}
    return pd.DataFrame([row], columns=cols)


def decide(pd_value: float, approve_th: float, reject_th: float) -> str:
    if pd_value < approve_th:
        return "Approve"
    if pd_value >= reject_th:
        return "Reject"
    return "Review"


def apply_policy_overrides(applicant: Dict[str, Any], model_decision: str) -> str:
    """
    Simple business guardrails for MVP.
    These rules are intentionally lightweight and easy to explain in demo/interviews.
    """
    income = applicant.get("income")
    delinquencies = applicant.get("delinquencies")
    utilization = applicant.get("utilization")
    history_length = applicant.get("history_length")

    try:
        income = float(income) if income is not None else None
    except (TypeError, ValueError):
        income = None

    try:
        delinquencies = float(delinquencies) if delinquencies is not None else None
    except (TypeError, ValueError):
        delinquencies = None

    try:
        utilization = float(utilization) if utilization is not None else None
    except (TypeError, ValueError):
        utilization = None

    try:
        history_length = float(history_length) if history_length is not None else None
    except (TypeError, ValueError):
        history_length = None

    # Rule 1: extremely low income should not auto-approve
    if income is not None and income < 1000:
        return "Review" if model_decision == "Approve" else model_decision

    # Rule 2: very high utilization + any delinquency => reject
    if utilization is not None and delinquencies is not None:
        if utilization >= 0.95 and delinquencies >= 1:
            return "Reject"

    # Rule 3: very short credit history + repeated delinquencies => at least review
    if history_length is not None and delinquencies is not None:
        if history_length < 0.5 and delinquencies >= 2:
            return "Review" if model_decision == "Approve" else model_decision

    return model_decision


def score_one(applicant: Dict[str, Any]) -> Dict[str, Any]:
    bundle = get_model_bundle()

    approve_th, reject_th, thresholds = _extract_thresholds(bundle.metadata)

    X = applicant_to_df(applicant, bundle)
    proba = bundle.model.predict_proba(X)
    pd_value = float(proba[0][1])  # probability of class 1 (default)
    model_decision = decide(pd_value, approve_th, reject_th)
    final_decision = apply_policy_overrides(applicant, model_decision)

    return {
        "pd": pd_value,
        "decision": final_decision,
        "model_version": str(bundle.metadata.get("version", "unknown")),
        "thresholds": thresholds,
    }


def score_batch(applicants: List[Dict[str, Any]]) -> Dict[str, Any]:
    bundle = get_model_bundle()
    approve_th, reject_th, thresholds = _extract_thresholds(bundle.metadata)

    cols = bundle.features + bundle.segments
    df = pd.DataFrame(applicants)
    # ensure required columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]

    probas = bundle.model.predict_proba(df)[:, 1]
    results = []
    for i, p in enumerate(probas):
        p = float(p)
        model_decision = decide(p, approve_th, reject_th)
        final_decision = apply_policy_overrides(applicants[i], model_decision)
        results.append({"index": i, "pd": p, "decision": final_decision})

    return {
        "model_version": str(bundle.metadata.get("version", "unknown")),
        "thresholds": thresholds,
        "results": results,
    }
