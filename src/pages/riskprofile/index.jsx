import React from 'react'; // Removed unused useState, useMemo
import { Helmet } from 'react-helmet';
import Header from '../../components/Header';
import Footer from '../../components/Footer';
import NewsletterSection from './components/NewsletterSection';
import ModelDemo from './components/ModelDemo';
import ModelDemo2 from './components/ModelDemo2';
import Hero from './components/Hero';

const ResearchPublications = () => {
  // No logic needed here since we removed the publications part

  return (
    <>
      <Helmet>
        <title>Patient Risk Inference | OncoRisk</title>
        <meta name="description" content="Patient Risk Profile Inference" />
      </Helmet>
      
      <div className="min-h-screen bg-background">
        <Header />

        <main className="pt-20 lg:pt-24">
          
          {/* Top Section with DNA & Explainer */}
          <Hero />

          {/* Interactive Model Section */}
          <section className="bg-muted/30 border-y border-border">
            <div className="container mx-auto px-4 lg:px-8 py-10 md:py-14">
              <ModelDemo />
            </div>
          </section>
          {/* <ModelDemo2 /> */}

          {/* Newsletter / Contact Section */}
          <section className="container mx-auto px-4 lg:px-8 py-8 md:py-12">
            <NewsletterSection />
            
          </section>

        </main>

        <Footer />
      </div>
    </>
  );
};

export default ResearchPublications;