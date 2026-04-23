interface Props {
  message: string;
}

export default function ErrorState({ message }: Props) {
  return <div className="state-card state-card--error">Error: {message}</div>;
}
