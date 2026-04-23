import type { ApplicantInput } from "../api/types";

export const NAV_ITEMS = [
  { label: "Overview", path: "/" },
  { label: "Applicant Scoring", path: "/score" },
  { label: "Portfolio", path: "/portfolio" }
];

export const DEFAULT_APPLICANT: ApplicantInput = {
  age: 32,
  income: 68000,
  employment_length: 4,
  dti: 0.28,
  utilization: 0.42,
  delinquencies: 0,
  history_length: 7,
  tx_30d_count: 25,
  refund_rate_30d: 0.03,
  active_days_30d: 18,
  channel: "organic",
  region: "NE",
  product: "card"
};

export const PORTFOLIO_GROUP_OPTIONS = ["region", "channel", "product"];

export const CHANNEL_OPTIONS = ["organic", "paid_ads", "partner"];
export const REGION_OPTIONS = ["NE", "SE", "MW", "W"];
export const PRODUCT_OPTIONS = ["card", "installment", "bnpl"];
