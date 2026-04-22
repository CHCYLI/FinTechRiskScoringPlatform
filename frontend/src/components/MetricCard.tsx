interface Props {
  title: string;
  value: string;
  subtitle?: string;
}

export default function MetricCard({ title, value, subtitle }: Props) {
  return (
    <div className="card metric-card">
      <div className="metric-card__title">{title}</div>
      <div className="metric-card__value">{value}</div>
      {subtitle ? <div className="metric-card__subtitle">{subtitle}</div> : null}
    </div>
  );
}
