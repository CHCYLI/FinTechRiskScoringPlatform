import { PORTFOLIO_GROUP_OPTIONS } from "../lib/constants";

interface Props {
  value: string;
  onChange: (value: string) => void;
}

export default function PortfolioControls({ value, onChange }: Props) {
  return (
    <div className="card controls-row">
      <label htmlFor="groupBy">Group by</label>
      <select id="groupBy" value={value} onChange={(e) => onChange(e.target.value)}>
        {PORTFOLIO_GROUP_OPTIONS.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}
