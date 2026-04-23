import type { ScoreResponse } from "../api/types";
import { formatPercent } from "../lib/format";
import DecisionBadge from "./DecisionBadge";

interface Props {
  result: ScoreResponse;
}

export default function ScoreResultCard({ result }: Props) {
  return (
    <div className="card">
      <h3>Scoring Result</h3>
      <div className="result-grid">
        <div>
          <div className="muted-label">PD</div>
          <div className="result-value">{formatPercent(result.pd)}</div>
        </div>
        <div>
          <div className="muted-label">Decision</div>
          <DecisionBadge decision={result.decision} />
        </div>
        <div>
          <div className="muted-label">Model Version</div>
          <div className="result-text">{result.model_version}</div>
        </div>
      </div>
    </div>
  );
}
