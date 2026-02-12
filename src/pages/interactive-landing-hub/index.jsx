import React, { useEffect } from 'react';
import Header from '../../components/Header';
import Footer from '../../components/Footer';
import HeroSection from './components/HeroSection';
import TechnicalDeepDive from './components/TechnicalDeepDive';
import Cc from './components/Cc';
// import PerformanceMetrics from './components/PerformanceMetrics';
// import LandingPage from './components/LandingPage';

// import ProjectImpact from './components/ProjectImpact';
// import ModelDemo from './components/ModelDemo';
// import CaseStudies from './components/CaseStudies';
// import CTASection from './components/CTASection';

const InteractiveLandingHub = () => {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="pt-16 lg:pt-20">
       {/* <LandingPage /> */}
        <HeroSection />
        <Cc/>
        {/* <PerformanceMetrics /> */}
        {/* <ModelDemo /> */}
        {/* <CaseStudies /> */}
        {/* <CTASection /> */}
        <TechnicalDeepDive />
        {/* <ProjectImpact /> */}
      </main>
      <Footer />
    </div>
  );
};

export default InteractiveLandingHub;