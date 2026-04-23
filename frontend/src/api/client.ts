import axios from "axios";
import type {
  ApplicantInput,
  ExplainResponse,
  MetricsResponse,
  ModelVersionResponse,
  PortfolioSummaryResponse,
  ScoreResponse
} from "./types";

const envBase = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/v1";
const baseURL = envBase.endsWith("/v1") ? envBase : `${envBase}/v1`;

const api = axios.create({
  baseURL,
  timeout: 10000
});

export async function fetchModelVersion(): Promise<ModelVersionResponse> {
  const { data } = await api.get("/model/version");
  return data;
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const { data } = await api.get("/metrics");
  return data;
}

export async function scoreApplicant(payload: ApplicantInput): Promise<ScoreResponse> {
  const { data } = await api.post("/score", payload);
  return data;
}

export async function explainApplicant(payload: ApplicantInput): Promise<ExplainResponse> {
  const { data } = await api.post("/explain", payload);
  return data;
}

export async function fetchPortfolioSummary(
  groupBy: string
): Promise<PortfolioSummaryResponse> {
  const { data } = await api.get("/portfolio/summary", {
    params: { group_by: groupBy }
  });
  return data;
}
