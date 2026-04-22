import type { Decision } from "../api/types";

interface Props {
  decision: Decision;
}

export default function DecisionBadge({ decision }: Props) {
  return (
    <span className={`decision-badge decision-badge--${decision.toLowerCase()}`}>
      {decision}
    </span>
  );
}
