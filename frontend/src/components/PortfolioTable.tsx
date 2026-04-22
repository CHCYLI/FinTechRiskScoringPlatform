import type { PortfolioRow } from "../api/types";
import { formatPercent } from "../lib/format";

interface Props {
  rows: PortfolioRow[];
}

export default function PortfolioTable({ rows }: Props) {
  return (
    <div className="card">
      <h3>Portfolio Summary</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Group</th>
              <th>Count</th>
              <th>Avg PD</th>
              <th>Approve Rate</th>
              <th>Review Rate</th>
              <th>Reject Rate</th>
              <th>Bad Rate</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={`${row.group}-${idx}`}>
                <td>{row.group}</td>
                <td>{row.count}</td>
                <td>{formatPercent(row.avg_pd)}</td>
                <td>{formatPercent(row.approve_rate)}</td>
                <td>{formatPercent(row.review_rate)}</td>
                <td>{formatPercent(row.reject_rate)}</td>
                <td>{row.bad_rate == null ? "-" : formatPercent(row.bad_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
