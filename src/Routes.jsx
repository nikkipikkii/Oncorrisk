import React from "react";
import { BrowserRouter, Routes as RouterRoutes, Route } from "react-router-dom";
import ScrollToTop from "components/ScrollToTop";
import ErrorBoundary from "components/ErrorBoundary";
import NotFound from "pages/NotFound";
import DevelopmentPerformance from './pages/development-performance';
import InteractiveLandingHub from './pages/interactive-landing-hub';
import ClinicalImplementation from './pages/gene-db';
import ResearchPublications from './pages/riskprofile';
// analytics
import AnalyticsPage from './pages/analytics';
const Routes = () => {
  return (
    <BrowserRouter>
      <ErrorBoundary>
      <ScrollToTop />
      <RouterRoutes>
        {/* Define your route here */}
        <Route path="/" element={<InteractiveLandingHub />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/development-performance" element={<DevelopmentPerformance />} />
        <Route path="/interactive-landing-hub" element={<InteractiveLandingHub />} />
        <Route path="/clinical-implementation" element={<ClinicalImplementation />} />
        <Route path="/riskprofile" element={<ResearchPublications />} />
        <Route path="*" element={<NotFound />} />
      </RouterRoutes>
      </ErrorBoundary>
    </BrowserRouter>
  );
};

export default Routes;
