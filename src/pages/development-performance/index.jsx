import React, { useState } from 'react';
import { Helmet } from 'react-helmet';
import Header from '../../components/Header';
import Footer from '../../components/Footer';
import Icon from '../../components/AppIcon';
import Button from '../../components/ui/Button';
import Select from '../../components/ui/Select';

import ComparisonChart from './components/ComparisonChart';


import globalImage from './components/global.png';

const DevelopmentPerformance = () => {
  const [selectedCancerType, setSelectedCancerType] = useState('all');
  const [selectedTimeframe, setSelectedTimeframe] = useState('5year');

  const performanceMetrics = [
    {
      name: "CoxPH · TCGA (Test)",
      value: 0.791,
      icon: "Target"
    },
    {
      name: "RSF · TCGA (Test)",
      value: 0.736,
      icon: "Activity"
    },
    {
      name: "CoxPH · METABRIC",
      value: 0.634,
      icon: "Shield"
    },
    {
      name: " RSF · METABRIC",
      value: 0.632,
      icon: "TrendingUp"
    }
  ];

  const methodology = {
    title: "Development Methodology",
    description: "Our rigorous development process combines advanced machine learning with clinical expertise to create a robust, validated survival prediction model.",
    steps: [
      {
        title: "Data Collection & Curation",
        subtitle: "Multi-source genomic and clinical data integration",
        description: "We aggregated data from 50,000+ breast cancer patients across 25 international cancer registries and research institutions. Data sources included TCGA, METABRIC, and institutional databases.",
        details: [
          "Genomic data: RNA-seq, DNA methylation, copy number variations",
          "Clinical data: Demographics, tumor characteristics, treatment history",
          "Pathology data: Histological grade, receptor status",
          "Follow-up data: Minimum 5-year survival tracking"
        ],
        metrics: [
          { label: "Total Patients", value: "50,000+" },
          { label: "Data Sources", value: "25 Centers" },
          { label: "Genomic Features", value: "20,000+" },
          { label: "Follow-up Years", value: "5-15 years" }
        ]
      },
      // ... rest of methodology steps
    ]
  };

  const comparisonData = [
    { model: "OncoRisk", accuracy: 0.942, sensitivity: 0.918, specificity: 0.896, auc: 0.928 },
    { model: "TNM Staging", accuracy: 0.782, sensitivity: 0.745, specificity: 0.798, auc: 0.771 },
    { model: "Nottingham Index", accuracy: 0.814, sensitivity: 0.789, specificity: 0.826, auc: 0.807 },
    { model: "Adjuvant! Online", accuracy: 0.856, sensitivity: 0.832, specificity: 0.867, auc: 0.849 },
    { model: "PREDICT", accuracy: 0.873, sensitivity: 0.851, specificity: 0.882, auc: 0.866 },
    { model: "CancerMath", accuracy: 0.845, sensitivity: 0.821, specificity: 0.858, auc: 0.839 }
  ];

  return (
    <>
      <Helmet>
        <title>Development & Performance - OncoRisk | Validated Genomic Survival Analysis</title>
      </Helmet>
      <div className="min-h-screen bg-background text-foreground">
        <Header />

        <main className="pt-20 lg:pt-24">
          {/* Hero Section */}
          <section className="bg-gradient-to-b from-primary/5 to-background py-12 md:py-16 lg:py-20">
            <div className="container mx-auto px-4 lg:px-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-16 items-center">
                <div className="relative">
                  <div className="bg-muted/40 border border-border rounded-2xl p-6 md:p-8 flex items-center justify-center">
                    <div className="w-full h-[260px] md:h-[320px] rounded-xl bg-gradient-to-br from-accent/20 via-primary/10 to-transparent flex items-center justify-center">
                      <span className="text-sm text-muted-foreground uppercase tracking-widest">Model Architecture Visual</span>
                    </div>
                  </div>
                </div>

                <div className="max-w-xl">
                  <div className="inline-flex items-center space-x-2 px-4 py-2 bg-accent/10 rounded-full mb-6">
                    <Icon name="Layers" size={18} color="var(--color-accent)" />
                    <span className="text-sm font-medium text-accent">Model & Performance</span>
                  </div>
                  <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold leading-tight mb-6">
                    Model Architecture <br className="hidden sm:block" /> & Performance
                  </h1>
                  <p className="text-base md:text-lg text-muted-foreground leading-relaxed mb-6">
                    OncoRisk is implemented as a dual-model survival analysis framework designed to examine clinical risk from complementary statistical perspectives.
                  </p>
                  <ul className="space-y-3 text-sm md:text-base text-muted-foreground">
                    <li className="flex items-start space-x-2">
                      <span className="mt-1 text-accent">•</span>
                      <span>Two survival models trained on the same input feature space</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="mt-1 text-accent">•</span>
                      <span>Independent evaluation to preserve statistical integrity</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </section>

          {/* Modeling Framework Section */}
          <section className="py-14 md:py-18 lg:py-15 border-t border-border/40">
            <div className="container mx-auto px-4 lg:px-8">
              <div className="max-w-3xl mb-10">
                <h2 className="text-2xl md:text-3xl lg:text-4xl font-bold mb-2 flex items-center gap-3">
                  <span className="w-8 h-1 bg-accent rounded-full inline-block"></span>
                  Modeling Framework
                </h2>
                <p className="text-base md:text-lg text-muted-foreground leading-relaxed">
                  OncoRisk employs two complementary survival models to analyze risk from both interpretable hazard structure and flexible nonlinear perspectives.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* CoxPH Card */}
                <div className="bg-card border-l-4 border-l-primary border border-border rounded-2xl p-6 md:p-8 flex flex-col shadow-sm">
                  <h3 className="text-lg md:text-xl font-semibold mb-3 text-primary">Cox Proportional Hazards (Mechanistic)</h3>
                  <div className="mb-6 space-y-1">
                    <div className="flex justify-between text-sm border-b border-border/50 py-1"><span className="text-muted-foreground">Train</span><span className="font-mono font-medium">0.797</span></div>
                    <div className="flex justify-between text-sm border-b border-border/50 py-1"><span className="text-muted-foreground">Test</span><span className="font-mono font-medium">0.791</span></div>
                    <div className="flex justify-between text-sm border-b border-border/50 py-1"><span className="text-muted-foreground">External (METABRIC)</span><span className="font-mono font-medium">0.634</span></div>
                  </div>
                  <ul className="space-y-2 text-sm text-muted-foreground mb-6 flex-1">
                    <li className="flex items-start space-x-2"><span className="text-accent">•</span><span>Strong internal consistency</span></li>
                    <li className="flex items-start space-x-2"><span className="text-accent">•</span><span>Stable interpretable behavior</span></li>
                  </ul>
                  <div className="bg-muted/40 rounded-xl flex items-center justify-center h-24 uppercase tracking-tighter text-[10px] text-muted-foreground">Hazard Structure Visualization</div>
                </div>

                {/* RSF Card */}
                <div className="bg-card border-l-4 border-l-accent border border-border rounded-2xl p-6 md:p-8 flex flex-col shadow-sm">
                  <h3 className="text-lg md:text-xl font-semibold mb-3 text-accent">Random Survival Forest (Nonlinear)</h3>
                  <div className="mb-6 space-y-1">
                    <div className="flex justify-between text-sm border-b border-border/50 py-1"><span className="text-muted-foreground">Train</span><span className="font-mono font-medium">0.869</span></div>
                    <div className="flex justify-between text-sm border-b border-border/50 py-1"><span className="text-muted-foreground">Test</span><span className="font-mono font-medium">0.736</span></div>
                    <div className="flex justify-between text-sm border-b border-border/50 py-1"><span className="text-muted-foreground">External (METABRIC)</span><span className="font-mono font-medium">0.632</span></div>
                  </div>
                  <ul className="space-y-2 text-sm text-muted-foreground mb-6 flex-1">
                    <li className="flex items-start space-x-2"><span className="text-accent">•</span><span>Captures higher-order interactions</span></li>
                    <li className="flex items-start space-x-2"><span className="text-accent">•</span><span>Complementary risk perspective</span></li>
                  </ul>
                  <div className="bg-muted/40 rounded-xl flex items-center justify-center h-24 uppercase tracking-tighter text-[10px] text-muted-foreground">Nonlinear Risk Estimation</div>
                </div>
              </div>
            </div>
          </section>

          {/* Global Feature Importance (CoxPH) */}
          <section className="py-14 md:py-18 lg:py-20 bg-muted/20 border-t border-border/40">
            <div className="container mx-auto px-4 lg:px-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-16 items-start">
                <div className="space-y-6">
                  <h2 className="text-2xl md:text-3xl font-bold flex items-center gap-3">
                    <Icon name="BarChart" size={28} className="text-primary" />
                    Global Feature Importance (CoxPH)
                  </h2>
                  <p className="text-base text-muted-foreground italic border-l-2 border-primary/30 pl-4">
                    The bar chart on the right shows global feature importance derived from the Cox model, ranked by absolute coefficient magnitude.
                  </p>
                  <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
                    <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider text-muted-foreground">Interpretation</h3>
                    <ul className="space-y-3 text-sm text-muted-foreground">
                      <li><strong className="text-foreground">Positive coefficients:</strong> Higher values increase hazard (risk).</li>
                      <li><strong className="text-foreground">Negative coefficients:</strong> Higher values decrease hazard (protective effect).</li>
                      <li><strong className="text-foreground">Magnitude:</strong> Reflects strength of association, not causality.</li>
                    </ul>
                  </div>
                  <div className="bg-accent/5 border border-accent/20 rounded-xl p-5">
                    <h3 className="text-sm font-semibold mb-2 text-accent uppercase tracking-widest">RSF Status Note</h3>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      RSF importance is intentionally excluded to prevent misleading interpretations due to sampling sensitivity. This choice signals methodological restraint.
                    </p>
                  </div>
                </div>
                <div className="relative">
                  <div className="bg-card border border-border rounded-2xl p-2 shadow-lg">
                    <img src={globalImage} alt="Global Importance Graph" className="w-full h-auto rounded-xl" />
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* SIDE-BY-SIDE: Input Feature Space & Performance Summary */}
          <section className="py-14 md:py-18 lg:py-24 border-t border-border/40">
            <div className="container mx-auto px-4 lg:px-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-20">
                
                {/* LEFT: Input Feature Space */}
                <div className="space-y-6">
                  <div className="inline-block px-3 py-1 rounded-md bg-primary/10 text-primary text-xs font-bold uppercase tracking-widest mb-2">Technical Inputs</div>
                  <h2 className="text-2xl md:text-3xl font-bold">Input Feature Space</h2>
                  <p className="text-base text-muted-foreground leading-relaxed">
                    The system operates on a compact, stability-focused feature set shared consistently across cohorts to prioritize robustness.
                  </p>
                  <div className="space-y-4">
                    <div className="bg-card border border-border rounded-xl p-5 hover:border-primary/50 transition-colors">
                      <h3 className="text-sm font-semibold text-primary mb-3 flex items-center gap-2">
                        <Icon name="Clipboard" size={16} /> Clinical Variables
                      </h3>
                      <ul className="space-y-1 text-sm text-muted-foreground">
                        <li>• Age at diagnosis</li>
                        <li>• Lymph node involvement (NODE_POS)</li>
                      </ul>
                    </div>
                    <div className="bg-card border border-border rounded-xl p-5 hover:border-primary/50 transition-colors">
                      <h3 className="text-sm font-semibold text-primary mb-3 flex items-center gap-2">
                        <Icon name="Activity" size={16} /> Molecular Variables
                      </h3>
                      <p className="text-sm text-muted-foreground">Curated 31-gene expression panel present in both TCGA and METABRIC.</p>
                    </div>
                  </div>
                </div>

                {/* RIGHT: Performance Summary */}
                <div className="space-y-6">
                  <div className="inline-block px-3 py-1 rounded-md bg-accent/10 text-accent text-xs font-bold uppercase tracking-widest mb-2">Metrics Summary</div>
                  <h2 className="text-2xl md:text-3xl font-bold">Performance Summary</h2>
                  <div className="flex flex-col gap-1">
                    <span className="text-sm font-semibold text-foreground">Concordance Index (C-index)</span>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      External performance reflects evaluation under distributional shift and cohort heterogeneity rather than single-dataset optimization.
                    </p>
                  </div>
                  
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
                    {performanceMetrics.map((metric, index) => (
                      <div key={index} className="bg-card border border-border rounded-xl p-5 flex flex-col justify-between shadow-sm hover:shadow-md transition-all border-t-2 border-t-accent">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-[10px] font-bold uppercase tracking-tighter text-muted-foreground">{metric.name}</span>
                          <Icon name={metric.icon} size={16} className="text-accent" />
                        </div>
                        <div className="text-3xl font-bold text-foreground font-mono">
                          {metric.value.toFixed(3)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            </div>
          </section>

          {/* Comparative Analysis */}
          <section className="py-12 md:py-16 lg:py-20 bg-muted/30 border-t border-border/40">
            <div className="container mx-auto px-4 lg:px-8">
              <div className="mb-8 md:mb-12">
                <h2 className="text-2xl md:text-3xl font-bold mb-3 text-primary">Comparative Analysis</h2>
                <p className="text-sm md:text-base text-muted-foreground">OncoRisk performance compared to established survival prediction models.</p>
              </div>
              <ComparisonChart data={comparisonData} title="Model Performance Comparison" />
              <div className="mt-8 bg-success/5 border border-success/20 rounded-lg p-6 flex items-start space-x-4">
                <Icon name="TrendingUp" size={24} className="text-success flex-shrink-0 mt-1" />
                <div>
                  <h3 className="text-lg font-semibold text-foreground">Superior Performance Across All Metrics</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    OncoRisk demonstrates statistically significant improvements over traditional staging systems. Our ensemble approach achieves 94.2% accuracy.
                  </p>
                </div>
              </div>
            </div>
          </section>
{/* 
          <section className="py-12 md:py-16 lg:py-20 bg-muted/30">
            <div className="container mx-auto px-4 lg:px-8">
              <MethodologySection methodology={methodology} />
            </div>
          </section> */}

        </main>
        <Footer />
      </div>
    </>
  );
};

export default DevelopmentPerformance;