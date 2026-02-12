import React, { useState, useEffect } from 'react';
import Icon from '../../../components/AppIcon';


// Sub-component for the Expandable Sections
const ExpandableSection = ({ title, icon, summary, children, isOpen, onToggle, colorClass }) => {
  return (
    <div className={`border rounded-2xl mb-6 transition-all duration-300 overflow-hidden ${
      isOpen ? 'bg-card border-border shadow-lg' : 'bg-card/50 border-border/50 hover:bg-card hover:border-border'
    }`}>
      {/* Header (Always Visible) */}
      <div 
        onClick={onToggle}
        className="p-6 md:p-8 cursor-pointer group"
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-4">
            <div className={`mt-1 p-3 rounded-xl ${colorClass} bg-opacity-10 text-opacity-100`}>
              <Icon name={icon} size={24} className={colorClass.replace('bg-', 'text-')} />
            </div>
            <div>
              <h3 className="text-xl md:text-2xl font-bold text-foreground mb-3 group-hover:text-primary transition-colors">
                {title}
              </h3>
              {/* Core Summary (Always Visible) */}
              <p className="text-base text-muted-foreground leading-relaxed max-w-3xl">
                {summary}
              </p>
            </div>
          </div>
          
          {/* Chevron Indicator */}
          <div className={`p-2 rounded-full border transition-all duration-300 mt-1 ${
            isOpen ? 'bg-primary text-primary-foreground border-primary rotate-180' : 'bg-transparent border-border text-muted-foreground group-hover:border-primary group-hover:text-primary'
          }`}>
            <Icon name="ChevronDown" size={20} />
          </div>
        </div>
        
        {!isOpen && (
            <div className="mt-4 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-primary opacity-0 group-hover:opacity-100 transition-opacity pl-[4.5rem]">
                <span>Read Technical Details</span>
                <Icon name="ArrowRight" size={12} />
            </div>
        )}
      </div>

      {/* Expanded Content */}
      <div 
        className={`grid transition-[grid-template-rows] duration-500 ease-out ${
          isOpen ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'
        }`}
      >
        <div className="overflow-hidden">
          <div className="px-6 md:px-8 pb-8 pt-0 pl-[calc(1.5rem+3rem)] md:pl-[calc(2rem+3rem)]">
            <div className="h-px w-full bg-border mb-8"></div>
            {children}
          </div>
        </div>
      </div>
    </div>
  );
};

// Component named 'cc' as requested
const Cc = () => {
  // Track which section is open (null = all closed)
  const [openSection, setOpenSection] = useState('architecture');

  const toggleSection = (id) => {
    setOpenSection(openSection === id ? null : id);
  };

  return (
    <section className="py-20 md:py-32 bg-background relative overflow-hidden">
      {/* Background Decor */}
      <div className="absolute top-0 right-0 w-1/3 h-1/3 bg-primary/5 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-1/3 h-1/3 bg-accent/5 rounded-full blur-[100px] pointer-events-none" />

      <div className="container mx-auto px-4 md:px-6">
        
        {/* Section Header */}
        <div className="mb-16 md:mb-20 max-w-4xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-bold uppercase tracking-widest mb-6">
            <Icon name="Cpu" size={14} />
            <span>Technical Deep Dive</span>
          </div>
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-foreground mb-6">
            Dual-Model System for Robust Survival Prediction
          </h2>
          <p className="text-lg text-muted-foreground">
            Explore the architecture, data provenance, and scope of the OncoRisk system. 
            Click the sections below to expand technical specifications.
          </p>
        </div>

        {/* ACCORDION 1: ARCHITECTURE */}
        <ExpandableSection
          title="How the System Works"
          icon="GitMerge"
          colorClass="bg-blue-500 text-blue-500"
          isOpen={openSection === 'architecture'}
          onToggle={() => toggleSection('architecture')}
          summary="OncoRisk is built as a dual-model survival system where two complementary frameworks interrogate risk from different structural perspectives. Rather than competing for performance, they act as a dual lens for stability."
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            {/* Primary Model */}
            <div className="bg-secondary/20 rounded-xl p-6 border border-border">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-lg font-bold text-foreground flex items-center gap-2">
                  <Icon name="Activity" size={18} className="text-primary" />
                  Primary: Cox Proportional Hazards
                </h4>
                <span className="text-xs font-mono bg-primary/10 text-primary px-2 py-1 rounded">Mechanistic</span>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                Selected for its long-standing role in clinical survival analysis and explicit coefficient structure.
              </p>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <Icon name="Check" size={16} className="text-green-500 mt-0.5" />
                  <span>Derives relative hazard estimates</span>
                </li>
                <li className="flex items-start gap-2">
                  <Icon name="Check" size={16} className="text-green-500 mt-0.5" />
                  <span>Provides full survival curves & RMST</span>
                </li>
                <li className="flex items-start gap-2">
                  <Icon name="Check" size={16} className="text-green-500 mt-0.5" />
                  <span>Interpretive backbone of the system</span>
                </li>
              </ul>
            </div>

            {/* Secondary Model */}
            <div className="bg-secondary/20 rounded-xl p-6 border border-border">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-lg font-bold text-foreground flex items-center gap-2">
                  <Icon name="GitBranch" size={18} className="text-accent" />
                  Secondary: Random Survival Forests
                </h4>
                <span className="text-xs font-mono bg-accent/10 text-accent px-2 py-1 rounded">Nonlinear</span>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                Trained on the same feature set to capture nonlinear effects and feature interactions that linear models miss.
              </p>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <Icon name="Check" size={16} className="text-green-500 mt-0.5" />
                  <span>Captures complex interactions</span>
                </li>
                <li className="flex items-start gap-2">
                  <Icon name="Check" size={16} className="text-green-500 mt-0.5" />
                  <span>Independent survival estimates</span>
                </li>
                <li className="flex items-start gap-2">
                  <Icon name="Check" size={16} className="text-green-500 mt-0.5" />
                  <span>Used as a stability check, not replacement</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="bg-primary/5 border border-primary/10 rounded-xl p-6">
            <h4 className="font-bold text-foreground mb-2 flex items-center gap-2">
              <Icon name="HelpCircle" size={18} className="text-primary" />
              Why Two Models?
            </h4>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Cox provides a mechanistic, interpretable structure, while RSF offers flexibility. 
              <strong> Agreement between the two is a signal of structural stability.</strong> Disagreement is surfaced explicitly as uncertainty, rather than being suppressed through model averaging.
            </p>
          </div>
        </ExpandableSection>

        {/* ACCORDION 2: DATASETS */}
      {/* ACCORDION 2: DATASETS */}
        <ExpandableSection
          title="Datasets and Input Design"
          icon="Database"
          colorClass="bg-purple-500 text-purple-500"
          isOpen={openSection === 'datasets'}
          onToggle={() => toggleSection('datasets')}
          summary="OncoRisk is constructed using two large, publicly available breast cancer cohorts with no sample overlap. Cohort separation is treated as a rigid design constraint to ensure rigorous external validation."
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            
            {/* TCGA-BRCA */}
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-border pb-2 mb-3">
                <h4 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">
                  Training & Internal Eval
                </h4>
                <span className="text-xs font-mono bg-secondary text-foreground px-2 py-0.5 rounded">TCGA-BRCA</span>
              </div>
              
              <div className="bg-secondary/10 p-4 rounded-xl border border-border/50">
                <p className="text-sm text-foreground font-medium mb-2">
                  The Cancer Genome Atlas (Genomically Profiled)
                </p>
                <p className="text-sm text-muted-foreground leading-relaxed mb-3">
                  A genomically profiled dataset used exclusively for model development. 
                  To prevent information leakage and preserve reproducibility, a <strong>frozen train–test split</strong> was enforced throughout the project. 
                  All model fitting, feature scaling, and parameter estimation were performed strictly on the training set.
                </p>
                
                <div className="grid grid-cols-2 gap-3 mt-4">
                  <div className="bg-background p-3 rounded-lg text-center border border-border/50">
                    <div className="text-lg font-bold text-primary">847</div>
                    <div className="text-[10px] text-muted-foreground uppercase tracking-wide font-semibold">Training Patients</div>
                    <div className="text-[10px] text-primary/80 mt-1">(120 Events)</div>
                  </div>
                  <div className="bg-background p-3 rounded-lg text-center border border-border/50">
                    <div className="text-lg font-bold text-accent">212</div>
                    <div className="text-[10px] text-muted-foreground uppercase tracking-wide font-semibold">Internal Test</div>
                    <div className="text-[10px] text-accent/80 mt-1">(30 Events)</div>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground italic mt-3 text-center">
                  Total: 1,059 patients with complete clinical & genomic data.
                </p>
              </div>
            </div>

            {/* METABRIC */}
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-border pb-2 mb-3">
                <h4 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">
                  External Validation
                </h4>
                <span className="text-xs font-mono bg-secondary text-foreground px-2 py-0.5 rounded">METABRIC</span>
              </div>

              <div className="bg-secondary/10 p-4 rounded-xl border border-border/50 h-full">
                 <p className="text-sm text-foreground font-medium mb-2">
                  Microarray-Based Long-Term Follow-up
                </p>
                <p className="text-sm text-muted-foreground leading-relaxed mb-3">
                  An independently collected cohort generated using microarray-based profiling. 
                  Its role is to test whether the survival structure learned from TCGA <strong>generalizes</strong> across differences in cohort composition, measurement technology, and clinical practice.
                </p>

                <div className="flex items-center gap-4 bg-green-500/10 p-3 rounded-lg border border-green-500/20 mt-4">
                  <div className="text-center min-w-[60px]">
                     <div className="text-lg font-bold text-green-600 dark:text-green-400">1,903</div>
                     <div className="text-[9px] text-green-600/70 dark:text-green-400/70 font-bold uppercase">Patients</div>
                  </div>
                  <div className="h-8 w-px bg-green-500/20"></div>
                  <div>
                    <div className="font-bold text-green-700 dark:text-green-400 text-sm">Strictly Independent</div>
                    <div className="text-xs text-green-600/80 dark:text-green-500/80">
                      1,103 observed events. No samples used for training, selection, or tuning.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="h-px bg-border w-full my-8"></div>

          {/* Feature Space */}
          <div>
            <h4 className="text-lg font-bold text-foreground mb-4 flex items-center gap-2">
              <Icon name="Layers" size={20} className="text-purple-500" />
              Input Feature Space (33 Total)
            </h4>
            <p className="text-sm text-muted-foreground mb-6 max-w-3xl">
              Gene expression was restricted to a curated <strong>31-gene signature</strong> present in both cohorts, enforcing cross-cohort compatibility and reducing overfitting risk.
            </p>
            <div className="flex flex-col md:flex-row gap-6">
               <div className="flex-1 bg-card border border-border rounded-xl p-5">
                  <div className="text-sm font-bold text-primary mb-3">Clinical (2 Features)</div>
                  <ul className="space-y-3">
                    <li className="text-sm text-muted-foreground flex items-start gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 shrink-0"></span> 
                        <span>
                            <strong className="text-foreground">Age at diagnosis:</strong> Baseline risk modifier.
                        </span>
                    </li>
                    <li className="text-sm text-muted-foreground flex items-start gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 shrink-0"></span> 
                        <span>
                            <strong className="text-foreground">Lymph-node involvement:</strong> Strongest clinical predictor.
                        </span>
                    </li>
                  </ul>
               </div>
               <div className="flex-[2] bg-card border border-border rounded-xl p-5">
                  <div className="text-sm font-bold text-accent mb-3">Molecular (31 Features)</div>
                  <p className="text-sm text-muted-foreground mb-4">
                    A curated gene expression panel harmonized across TCGA (RNA-Seq) and METABRIC (Microarray).
                  </p>
                  <div className="space-y-2">
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <span className="px-2 py-1 bg-secondary rounded border border-border">Immune Regulation</span>
                          <span className="h-px flex-1 bg-border"></span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <span className="px-2 py-1 bg-secondary rounded border border-border">Metabolic Plasticity</span>
                          <span className="h-px flex-1 bg-border"></span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <span className="px-2 py-1 bg-secondary rounded border border-border">ECM Remodeling</span>
                          <span className="h-px flex-1 bg-border"></span>
                      </div>
                  </div>
               </div>
            </div>
          </div>
        </ExpandableSection>

        {/* ACCORDION 3: SCOPE */}
        <ExpandableSection
          title="System Outputs & Intended Scope"
          icon="Target"
          colorClass="bg-orange-500 text-orange-500"
          isOpen={openSection === 'scope'}
          onToggle={() => toggleSection('scope')}
          summary="OncoRisk reports relative hazard, survival curves, and RMST to support research. It does NOT produce clinical treatment recommendations or diagnostic classifications."
        >
           <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Outputs */}
              <div>
                <h4 className="font-bold text-foreground mb-4">What the System Produces</h4>
                <div className="bg-secondary/20 rounded-xl p-1">
                    {[
                        "Relative Hazard Estimates",
                        "Nonlinear Risk Measures",
                        "Survival Curves (Time-to-event)",
                        "Median Survival Time",
                        "Restricted Mean Survival Time (RMST)",
                        "Agreement Score"
                    ].map((item, i) => (
                        <div key={i} className="flex items-center gap-3 p-3 border-b border-border/50 last:border-0">
                            <Icon name="BarChart2" size={16} className="text-muted-foreground" />
                            <span className="text-sm text-foreground">{item}</span>
                        </div>
                    ))}
                </div>
              </div>

              {/* Scope Table */}
              <div>
                <h4 className="font-bold text-foreground mb-4">Operational Boundaries</h4>
                <div className="space-y-4">
                    {/* IS */}
                    <div className="bg-green-500/5 border border-green-500/20 rounded-xl p-5">
                        <h5 className="text-green-600 dark:text-green-400 font-bold text-sm uppercase mb-3 flex items-center gap-2">
                            <Icon name="CheckCircle" size={16} /> What This Is
                        </h5>
                        <ul className="space-y-2">
                            <li className="text-sm text-muted-foreground">A dual-model survival framework</li>
                            <li className="text-sm text-muted-foreground">A clinicogenomic integration study</li>
                            <li className="text-sm text-muted-foreground">A platform for interpretable exploration</li>
                        </ul>
                    </div>

                    {/* IS NOT */}
                    <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-5">
                        <h5 className="text-red-600 dark:text-red-400 font-bold text-sm uppercase mb-3 flex items-center gap-2">
                            <Icon name="XCircle" size={16} /> What This Is NOT
                        </h5>
                        <ul className="space-y-2">
                            <li className="text-sm text-muted-foreground">A diagnostic tool</li>
                            <li className="text-sm text-muted-foreground">A clinical decision system</li>
                            <li className="text-sm text-muted-foreground">A substitute for clinical judgment</li>
                        </ul>
                    </div>
                </div>
              </div>
           </div>
           
           <div className="mt-8 p-4 border-l-4 border-orange-500 bg-secondary/20 italic text-muted-foreground text-sm">
             “With the system’s scope defined, the following sections describe how risk is modeled, interpreted, and biologically grounded.”
           </div>
        </ExpandableSection>

      </div>
    </section>
  );
};

export default Cc;