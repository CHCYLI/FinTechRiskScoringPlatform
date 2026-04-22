import { useState } from "react";
import { explainApplicant, scoreApplicant } from "../api/client";
import type { ApplicantInput, ExplainResponse, ScoreResponse } from "../api/types";
import ApplicantForm from "../components/ApplicantForm";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import ReasonsList from "../components/ReasonsList";
import ScoreResultCard from "../components/ScoreResultCard";
import { DEFAULT_APPLICANT } from "../lib/constants";

export default function ApplicantScoringPage() {
  const [scoreResult, setScoreResult] = useState<ScoreResponse | null>(null);
  const [explainResult, setExplainResult] = useState<ExplainResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(values: ApplicantInput) {
    try {
      setSubmitting(true);
      setError("");
      const [score, explain] = await Promise.all([
        scoreApplicant(values),
        explainApplicant(values)
      ]);
      setScoreResult(score);
      setExplainResult(explain);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Scoring failed");
      setScoreResult(null);
      setExplainResult(null);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="page-grid page-grid--two-col">
      <div>
        <h1>Applicant Scoring</h1>
        <p className="page-subtitle">Submit one applicant to get PD, decision, and reasons.</p>
        <ApplicantForm
          initialValues={DEFAULT_APPLICANT}
          onSubmit={handleSubmit}
          submitting={submitting}
        />
      </div>

      <div className="stack-gap">
        {submitting ? <LoadingState /> : null}
        {error ? <ErrorState message={error} /> : null}

        {!submitting && !error && !scoreResult ? (
          <EmptyState message="No scoring result yet. Submit an applicant to begin." />
        ) : null}

        {scoreResult ? <ScoreResultCard result={scoreResult} /> : null}

        {explainResult ? (
          <ReasonsList
            topFeatures={explainResult.top_features}
            reasons={explainResult.reasons}
          />
        ) : null}
      </div>
    </div>
  );
}
