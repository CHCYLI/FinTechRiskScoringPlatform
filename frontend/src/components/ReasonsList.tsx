interface Props {
  topFeatures?: string[];
  reasons?: string[];
}

export default function ReasonsList({ topFeatures = [], reasons = [] }: Props) {
  if (!reasons.length && !topFeatures.length) return null;

  return (
    <div className="card">
      <h3>Explainability</h3>

      {topFeatures.length ? (
        <>
          <div className="muted-label">Top Features</div>
          <ul>
            {topFeatures.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </>
      ) : null}

      {reasons.length ? (
        <>
          <div className="muted-label">Reasons</div>
          <ul>
            {reasons.map((r, i) => (
              <li key={`${r}-${i}`}>{r}</li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}
