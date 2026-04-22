import type { ReactNode } from "react";
import NavBar from "./NavBar";

interface Props {
  children: ReactNode;
}

export default function Layout({ children }: Props) {
  return (
    <div className="app-shell">
      <NavBar />
      <main className="page-container">{children}</main>
    </div>
  );
}
