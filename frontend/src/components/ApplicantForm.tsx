import { useState } from "react";
import type { ApplicantInput } from "../api/types";
import { CHANNEL_OPTIONS, PRODUCT_OPTIONS, REGION_OPTIONS } from "../lib/constants";

interface Props {
  initialValues: ApplicantInput;
  onSubmit: (values: ApplicantInput) => Promise<void> | void;
  submitting?: boolean;
}

const numericKeys: Array<keyof ApplicantInput> = [
  "age",
  "income",
  "employment_length",
  "dti",
  "utilization",
  "delinquencies",
  "history_length",
  "tx_30d_count",
  "refund_rate_30d",
  "active_days_30d"
];

export default function ApplicantForm({ initialValues, onSubmit, submitting }: Props) {
  const [form, setForm] = useState<ApplicantInput>(initialValues);

  function handleChange<K extends keyof ApplicantInput>(key: K, value: string) {
    setForm((prev) => ({
      ...prev,
      [key]: numericKeys.includes(key) ? Number(value) : value
    }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    await onSubmit(form);
  }

  return (
    <form className="card form-grid" onSubmit={handleSubmit}>
      <div className="form-grid__row">
        <label>Age</label>
        <input type="number" value={form.age} onChange={(e) => handleChange("age", e.target.value)} />
      </div>

      <div className="form-grid__row">
        <label>Income</label>
        <input type="number" value={form.income} onChange={(e) => handleChange("income", e.target.value)} />
      </div>

      <div className="form-grid__row">
        <label>Employment Length</label>
        <input
          type="number"
          step="0.1"
          value={form.employment_length}
          onChange={(e) => handleChange("employment_length", e.target.value)}
        />
      </div>

      <div className="form-grid__row">
        <label>DTI</label>
        <input type="number" step="0.01" value={form.dti} onChange={(e) => handleChange("dti", e.target.value)} />
      </div>

      <div className="form-grid__row">
        <label>Utilization</label>
        <input
          type="number"
          step="0.01"
          value={form.utilization}
          onChange={(e) => handleChange("utilization", e.target.value)}
        />
      </div>

      <div className="form-grid__row">
        <label>Delinquencies</label>
        <input
          type="number"
          value={form.delinquencies}
          onChange={(e) => handleChange("delinquencies", e.target.value)}
        />
      </div>

      <div className="form-grid__row">
        <label>History Length</label>
        <input
          type="number"
          step="0.1"
          value={form.history_length}
          onChange={(e) => handleChange("history_length", e.target.value)}
        />
      </div>

      <div className="form-grid__row">
        <label>Transactions (30d)</label>
        <input
          type="number"
          value={form.tx_30d_count}
          onChange={(e) => handleChange("tx_30d_count", e.target.value)}
        />
      </div>

      <div className="form-grid__row">
        <label>Refund Rate (30d)</label>
        <input
          type="number"
          step="0.01"
          value={form.refund_rate_30d}
          onChange={(e) => handleChange("refund_rate_30d", e.target.value)}
        />
      </div>

      <div className="form-grid__row">
        <label>Active Days (30d)</label>
        <input
          type="number"
          value={form.active_days_30d}
          onChange={(e) => handleChange("active_days_30d", e.target.value)}
        />
      </div>

      <div className="form-grid__row">
        <label>Channel</label>
        <select value={form.channel} onChange={(e) => handleChange("channel", e.target.value)}>
          {CHANNEL_OPTIONS.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>

      <div className="form-grid__row">
        <label>Region</label>
        <select value={form.region} onChange={(e) => handleChange("region", e.target.value)}>
          {REGION_OPTIONS.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>

      <div className="form-grid__row">
        <label>Product</label>
        <select value={form.product} onChange={(e) => handleChange("product", e.target.value)}>
          {PRODUCT_OPTIONS.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>

      <div className="form-grid__actions">
        <button type="submit" disabled={submitting}>
          {submitting ? "Scoring..." : "Score Applicant"}
        </button>
      </div>
    </form>
  );
}
