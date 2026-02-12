import React, { useEffect } from 'react';
import Header from '../../components/Header';
import Footer from '../../components/Footer';
import HeroSection from './components/HeroSection';
import GeneIntelligence from './components/GeneIntelligence';


const ClinicalImplementation = () => {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main>
        <HeroSection />
       
        <GeneIntelligence />

      </main>
      <Footer />
    </div>
  );
};

export default ClinicalImplementation;