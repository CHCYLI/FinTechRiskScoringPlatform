export type Decision = "Approve" | "Review" | "Reject";

export interface ApplicantInput {
  age: number;
  income: number;
  employment_length: number;
  dti: number;
  utilization: number;
  delinquencies: number;
  history_length: number;
  tx_30d_count: number;
  refund_rate_30d: number;
  active_days_30d: number;
  channel: string;
  region: string;
  product: string;
}

export interface ScoreResponse {
  pd: number;
  decision: Decision;
  model_version: string;
  thresholds?: Record<string, number>;
}

export interface ExplainResponse {
  model_version: string;
  top_features: string[];
  reasons: string[];
}

export interface ModelVersionResponse {
  status?: string;
  model_name?: string;
  version?: string;
  trained_at?: string;
  feature_schema_sha256?: string;
  features?: string[];
  segments?: string[];
  thresholds?: Record<string, number>;
  artifact_dir?: string;
  artifact_files?: Record<string, string>;
}

export interface MetricsResponse {
  status?: string;
  version?: string;
  trained_at?: string;
  metrics: Record<string, number> | Record<string, Record<string, number>> | null;
}

export interface PortfolioRow {
  group: string;
  count: number;
  avg_pd: number;
  approve_count?: number;
  review_count?: number;
  reject_count?: number;
  approve_rate?: number;
  review_rate?: number;
  reject_rate?: number;
  bad_rate?: number | null;
}

export interface PortfolioSummaryResponse {
  model_version?: string;
  group_by: string;
  filters?: Record<string, string | null>;
  rows: PortfolioRow[];
}
