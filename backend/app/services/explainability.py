from typing import Dict, List

from app.services.inference import get_model_bundle


FEATURE_REASON_MAP = {
    "utilization": "High credit utilization",
    "dti": "High debt-to-income ratio",
    "delinquencies": "Recent delinquencies increase default risk",
    "history_length": "Short credit history",
    "income": "Lower income may reduce repayment capacity",
    "employment_length": "Short employment length may indicate less income stability",
    "refund_rate_30d": "High refund rate may indicate unstable recent behavior",
    "active_days_30d": "Low recent activity may indicate weaker engagement",
    "tx_30d_count": "Low recent transaction volume may indicate weaker cash flow signal",
}


def explain_one_placeholder(applicant: Dict) -> Dict:
    bundle = get_model_bundle()
    version = str(bundle.metadata.get("version", "unknown"))

    scored_candidates: List[tuple[str, float]] = []

    utilization = applicant.get("utilization")
    if utilization is not None:
        scored_candidates.append(("utilization", float(utilization)))

    dti = applicant.get("dti")
    if dti is not None:
        scored_candidates.append(("dti", float(dti)))

    delinq = applicant.get("delinquencies")
    if delinq is not None:
        scored_candidates.append(("delinquencies", float(delinq)))

    history = applicant.get("history_length")
    if history is not None:
        scored_candidates.append(("history_length", -float(history)))

    income = applicant.get("income")
    if income is not None:
        scored_candidates.append(("income", -float(income)))

    emp_len = applicant.get("employment_length")
    if emp_len is not None:
        scored_candidates.append(("employment_length", -float(emp_len)))

    refund = applicant.get("refund_rate_30d")
    if refund is not None:
        scored_candidates.append(("refund_rate_30d", float(refund)))

    active = applicant.get("active_days_30d")
    if active is not None:
        scored_candidates.append(("active_days_30d", -float(active)))

    tx = applicant.get("tx_30d_count")
    if tx is not None:
        scored_candidates.append(("tx_30d_count", -float(tx)))

    ranked = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
    top_features = [name for name, _ in ranked[:3]]
    reasons = [FEATURE_REASON_MAP.get(f, f"Risk signal from {f}") for f in top_features]

    return {
        "model_version": version,
        "top_features": top_features,
        "reasons": reasons,
    }
