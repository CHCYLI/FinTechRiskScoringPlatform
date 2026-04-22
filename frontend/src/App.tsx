import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import ApplicantScoringPage from "./pages/ApplicantScoringPage";
import OverviewPage from "./pages/OverviewPage";
import PortfolioPage from "./pages/PortfolioPage";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/score" element={<ApplicantScoringPage />} />
        <Route path="/portfolio" element={<PortfolioPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
