import React from 'react';
import Header from '../../components/Header';
import Footer from '../../components/Footer';

import AnalyticsContent from './components/AnalyticsContent'; 

const AnalyticsPage = () => {
  return (
    <div className="min-h-screen bg-[#080c14] flex flex-col">
      <Header />
      
      <main className="flex-grow">
        <AnalyticsContent />
      </main>

      <Footer />
    </div>
  );
};

export default AnalyticsPage;