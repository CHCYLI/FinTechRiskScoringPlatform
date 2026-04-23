import { useEffect, useState } from "react";
import { fetchPortfolioSummary } from "../api/client";
import type { PortfolioSummaryResponse } from "../api/types";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import PortfolioControls from "../components/PortfolioControls";
import PortfolioTable from "../components/PortfolioTable";

export default function PortfolioPage() {
  const [groupBy, setGroupBy] = useState("region");
  const [data, setData] = useState<PortfolioSummaryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError("");
        const res = await fetchPortfolioSummary(groupBy);
        setData(res);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load portfolio");
        setData(null);
      } finally {
        setLoading(false);
      }
    }

    void load();
  }, [groupBy]);

  return (
    <div className="page-grid">
      <div>
        <h1>Portfolio Analytics</h1>
        <p className="page-subtitle">Aggregate portfolio risk by region, channel, or product.</p>
      </div>

      <PortfolioControls value={groupBy} onChange={setGroupBy} />

      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}

      {!loading && !error && data && data.rows.length > 0 ? (
        <PortfolioTable rows={data.rows} />
      ) : null}

      {!loading && !error && data && data.rows.length === 0 ? (
        <EmptyState message="No portfolio rows returned for the selected grouping." />
      ) : null}
    </div>
  );
}
