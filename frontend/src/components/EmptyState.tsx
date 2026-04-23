interface Props {
  message: string;
}

export default function EmptyState({ message }: Props) {
  return <div className="state-card">{message}</div>;
}
