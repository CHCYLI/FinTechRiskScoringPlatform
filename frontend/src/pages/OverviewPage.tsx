import { useEffect, useMemo, useState } from "react";
import { fetchMetrics, fetchModelVersion } from "../api/client";
import type { MetricsResponse, ModelVersionResponse } from "../api/types";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import MetricCard from "../components/MetricCard";
import { formatIsoDatetime, formatNumber } from "../lib/format";

function pickMetric(metrics: MetricsResponse | null, key: string): number | null {
  if (!metrics?.metrics) return null;

  const m = metrics.metrics as Record<string, unknown>;
  if (typeof m[key] === "number") return m[key] as number;

  const valBlock = m.val;
  if (valBlock && typeof valBlock === "object") {
    const nested = (valBlock as Record<string, unknown>)[key];
    if (typeof nested === "number") return nested;
  }
  return null;
}

export default function OverviewPage() {
  const [version, setVersion] = useState<ModelVersionResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError("");
        const [v, m] = await Promise.all([fetchModelVersion(), fetchMetrics()]);
        setVersion(v);
        setMetrics(m);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load overview");
      } finally {
        setLoading(false);
      }
    }
    void load();
  }, []);

  const valRocAuc = useMemo(() => pickMetric(metrics, "roc_auc"), [metrics]);
  const valPrAuc = useMemo(() => pickMetric(metrics, "pr_auc"), [metrics]);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState message={error} />;

  return (
    <div className="page-grid">
      <div>
        <h1>Overview</h1>
        <p className="page-subtitle">Model versioning and validation metrics snapshot.</p>
      </div>

      <div className="metrics-grid">
        <MetricCard title="Model Version" value={version?.version || "-"} />
        <MetricCard title="Model Name" value={version?.model_name || "-"} />
        <MetricCard title="Validation ROC-AUC" value={formatNumber(valRocAuc, 4)} />
        <MetricCard title="Validation PR-AUC" value={formatNumber(valPrAuc, 4)} />
      </div>

      <div className="card card-grid-two">
        <div>
          <div className="muted-label">Status</div>
          <div className="result-text">{version?.status || "-"}</div>
        </div>
        <div>
          <div className="muted-label">Trained At</div>
          <div className="result-text">{formatIsoDatetime(version?.trained_at)}</div>
        </div>
      </div>

      <div className="card">
        <h3>Model Metadata</h3>
        <pre className="json-block">{JSON.stringify(version, null, 2)}</pre>
      </div>

      <div className="card">
        <h3>Metrics</h3>
        <pre className="json-block">{JSON.stringify(metrics, null, 2)}</pre>
      </div>
    </div>
  );
}
