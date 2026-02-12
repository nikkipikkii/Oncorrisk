import React from 'react';
import { Link } from 'react-router-dom';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const AnalyticsContent = () => {

  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="text-white pb-24 font-sans selection:bg-cyan-500/30">
      
      {/* =========================================================================
          SECTION 1: HERO
         ========================================================================= */}
      <section className="relative pt-32 pb-24 overflow-hidden border-b border-white/5">
        <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-cyan-600/5 rounded-full blur-[120px] pointer-events-none"></div>
        <div className="container mx-auto px-4 md:px-6 lg:px-8 relative z-10">
          <div className="max-w-4xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-bold uppercase tracking-widest mb-6">
              <Icon name="BarChart2" size={14} />
              <span>Hierarchical Intelligence</span>
            </div>
            <h1 className="text-4xl md:text-6xl font-bold mb-6 tracking-tight">
              Survival <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
                Analytics
              </span>
            </h1>
            <p className="text-slate-400 text-lg md:text-xl leading-relaxed max-w-2xl">
              From broad population stratification to granular gene drivers. 
              OncoRisk provides a transparent, multi-level view of survival dynamics.
            </p>
          </div>
        </div>
      </section>

      {/* =========================================================================
          SECTION 2: THREE-COLUMN OVERVIEW (Linked)
         ========================================================================= */}
      <section className="py-16 bg-[#0f172a]/30 border-b border-white/5">
        <div className="container mx-auto px-4 md:px-6 lg:px-8">
          <div className="grid md:grid-cols-3 gap-6">
            
            {/* LEVEL 1: COHORT LINK */}
            <div 
              onClick={() => scrollToSection('cohort-analytics')}
              className="group relative bg-[#080c14] p-8 rounded-[2rem] border border-white/5 hover:border-cyan-500/30 transition-all hover:bg-[#0f172a] hover:scale-[1.01] cursor-pointer"
            >
              <div className="absolute top-6 right-6 opacity-10 group-hover:opacity-20 transition-opacity">
                <Icon name="Users" size={64} className="text-cyan-500" />
              </div>
              <div className="mb-6">
                <span className="text-slate-500 group-hover:text-cyan-400 font-bold tracking-widest text-xs uppercase bg-white/5 px-3 py-1 rounded-full border border-white/10 group-hover:border-cyan-500/20 transition-colors">
                  Level 01
                </span>
              </div>
              <h3 className="text-xl font-bold text-slate-300 group-hover:text-white mb-3 transition-colors">Cohort Analytics</h3>
              <p className="text-slate-500 group-hover:text-slate-400 text-sm leading-relaxed mb-6 transition-colors">
                Population-level risk stratification and validation via KM curves. 
              </p>
              <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-cyan-500 w-1/3 group-hover:w-full transition-all duration-700"></div>
              </div>
            </div>

            {/* LEVEL 2: FEATURE LINK */}
            <div 
              onClick={() => scrollToSection('feature-analytics')}
              className="group relative bg-[#080c14] p-8 rounded-[2rem] border border-white/5 hover:border-pink-500/30 transition-all hover:bg-[#0f172a] hover:scale-[1.01] cursor-pointer"
            >
              <div className="absolute top-6 right-6 opacity-10 group-hover:opacity-20 transition-opacity">
                <Icon name="Layers" size={64} className="text-pink-500" />
              </div>
              <div className="mb-6">
                <span className="text-slate-500 group-hover:text-pink-400 font-bold tracking-widest text-xs uppercase bg-white/5 px-3 py-1 rounded-full border border-white/10 group-hover:border-pink-500/20 transition-colors">
                  Level 02
                </span>
              </div>
              <h3 className="text-xl font-bold text-slate-300 group-hover:text-white mb-3 transition-colors">Feature Intelligence</h3>
              <p className="text-slate-500 group-hover:text-slate-400 text-sm leading-relaxed mb-6 transition-colors">
                Dual-model interpretation (Cox + RSF). Identifies variables that matter most.
              </p>
              <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-pink-500 w-1/3 group-hover:w-full transition-all duration-700"></div>
              </div>
            </div>

            {/* LEVEL 3: GENE LINK */}
            <div 
              onClick={() => scrollToSection('gene-analytics')}
              className="group relative bg-[#080c14] p-8 rounded-[2rem] border border-white/5 hover:border-emerald-500/30 transition-all hover:bg-[#0f172a] hover:scale-[1.01] cursor-pointer"
            >
              <div className="absolute top-6 right-6 opacity-10 group-hover:opacity-20 transition-opacity">
                <Icon name="Dna" size={64} className="text-emerald-500" />
              </div>
              <div className="mb-6">
                <span className="text-slate-500 group-hover:text-emerald-400 font-bold tracking-widest text-xs uppercase bg-white/5 px-3 py-1 rounded-full border border-white/10 group-hover:border-emerald-500/20 transition-colors">
                  Level 03
                </span>
              </div>
              <h3 className="text-xl font-bold text-slate-300 group-hover:text-white mb-3 transition-colors">Gene Intelligence</h3>
              <p className="text-slate-500 group-hover:text-slate-400 text-sm leading-relaxed mb-6 transition-colors">
                Granular drivers from the 31-gene panel with specific hazard coefficients.
              </p>
              <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-emerald-500 w-1/3 group-hover:w-full transition-all duration-700"></div>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* =========================================================================
          SECTION 3: DETAILED DEEP DIVE
         ========================================================================= */}
      <section className="container mx-auto px-4 md:px-6 lg:px-8 py-24">
        
        {/* --- 01. COHORT ANALYTICS --- */}
        <div id="cohort-analytics" className="mb-32 scroll-mt-32">
          {/* NUMBERED HEADER */}
          <div className="flex items-baseline gap-6 mb-12 border-b border-white/5 pb-8">
            <span className="text-6xl md:text-8xl font-black text-white/10 leading-none select-none">01</span>
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Cohort-Level Survival Structure</h2>
              <p className="text-slate-400 text-lg leading-relaxed max-w-3xl">
                Validating that the model meaningfully stratifies patients at the population level and generalizes across independent cohorts.
              </p>
            </div>
          </div>

          {/* CONTENT GRID */}
          <div className="grid lg:grid-cols-2 gap-16">
            <div className="space-y-8">
              <div className="space-y-4">
                <h3 className="text-xl font-bold text-white">Risk Stratification (TCGA & METABRIC)</h3>
                <p className="text-slate-400 leading-relaxed text-sm">
                   Patients are stratified into low-risk and high-risk groups based on the median Cox predicted hazard. 
                   Kaplan–Meier (KM) curves visualize time-to-event behavior to confirm:
                </p>
                <ul className="space-y-3 pt-2 text-sm text-slate-300">
                  <li className="flex gap-3">
                    <Icon name="Check" size={18} className="text-cyan-500" />
                    Distinct survival trajectories (High vs Low).
                  </li>
                  <li className="flex gap-3">
                    <Icon name="Check" size={18} className="text-cyan-500" />
                    Sustained divergence over time (not transient).
                  </li>
                  <li className="flex gap-3">
                    <Icon name="Check" size={18} className="text-cyan-500" />
                    Cross-cohort generalization 
                  </li>
                </ul>
              </div>
              
              <div className="p-6 bg-cyan-900/10 border-l-4 border-cyan-500 rounded-r-xl">
                 <p className="text-cyan-200 italic text-sm font-medium">
                   "Consistent separation in METABRIC supports cross-cohort generalization rather than dataset-specific overfitting."
                 </p>
              </div>
            </div>
            
            {/* GRAPH PLACEHOLDER */}
            <div className="bg-[#080c14] rounded-3xl border border-white/10 p-2 overflow-hidden shadow-2xl relative min-h-[300px] flex items-center justify-center group">
               <div className="absolute inset-0 bg-gradient-to-tr from-cyan-500/5 to-transparent pointer-events-none"></div>
               <div className="text-slate-600 font-mono text-sm group-hover:text-cyan-500 transition-colors">
                 
               </div>
            </div>
          </div>
        </div>

        {/* --- 02. FEATURE INTELLIGENCE --- */}
        <div id="feature-analytics" className="mb-32 scroll-mt-32">
          {/* NUMBERED HEADER */}
          <div className="flex items-baseline gap-6 mb-12 border-b border-white/5 pb-8">
            <span className="text-6xl md:text-8xl font-black text-white/10 leading-none select-none">02</span>
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Feature-Level Intelligence (Cox + RSF)</h2>
              <p className="text-slate-400 text-lg leading-relaxed max-w-3xl">
                Explaining which features the models rely on and how they influence risk differently across linear and nonlinear paradigms.
              </p>
            </div>
          </div>

          {/* CONTENT GRID */}
          <div className="grid lg:grid-cols-2 gap-12">
            
            {/* LEFT: COX */}
            <div className="space-y-6">
              <div className="p-6 bg-[#0f172a] border border-white/10 rounded-2xl">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-pink-500"></div>
                  Cox Proportional Hazards
                </h3>
                <p className="text-slate-400 text-sm mb-4 leading-relaxed">
                  Features are ranked by Cox coefficients. The x-axis is the log-hazard coefficient, reflecting direction and strength.
                </p>
                <div className="text-xs text-slate-500 space-y-2">
                  <p>• Positive Coef → Higher Risk</p>
                  <p>• Negative Coef → Protective Effect</p>
                </div>
              </div>
              
              <div className="bg-pink-900/10 border border-pink-500/10 rounded-2xl p-6">
                 <h4 className="text-pink-400 font-bold text-sm mb-2">Research Interpretation</h4>
                 <ul className="space-y-1 text-xs text-slate-400">
                   <li>• Use to sanity-check biological signal.</li>
                   <li>• Do NOT use to infer causality or make clinical claims.</li>
                 </ul>
              </div>
            </div>

            {/* RIGHT: RSF & SYNTHESIS */}
            <div className="space-y-6">
              <div className="p-6 bg-[#0f172a] border border-white/10 rounded-2xl">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                  Random Survival Forests (RSF)
                </h3>
                <p className="text-slate-400 text-sm mb-4 leading-relaxed">
                  RSF captures nonlinear effects and interactions. 
                  <span className="text-slate-500 block mt-2">
                    *RSF importance is intentionally withheld to avoid misleading interpretations of unstable nonlinear feature rankings.
                  </span>
                </p>
              </div>

              <div className="p-6 bg-blue-900/10 border border-blue-500/10 rounded-2xl">
                <h3 className="text-blue-400 font-bold text-sm mb-2">Why Two Models?</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Cox provides directional clarity. RSF provides structural flexibility. 
                  Agreement increases confidence; disagreement highlights uncertainty.
                </p>
              </div>
            </div>

          </div>
          
          {/* DISCLAIMER FOOTER */}
          <div className="mt-8 pt-6 border-t border-white/5 text-center">
            <p className="text-xs text-slate-500">
              Disclaimer: Feature importance reflects associations learned from population-level data and is provided for research interpretation only.
            </p>
          </div>
        </div>

        {/* --- 03. GENE INTELLIGENCE (LINKED) --- */}
        <div id="gene-analytics" className="scroll-mt-32">
          {/* NUMBERED HEADER */}
          <div className="flex items-baseline gap-6 mb-12 border-b border-white/5 pb-8">
            <span className="text-6xl md:text-8xl font-black text-white/10 leading-none select-none">03</span>
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Gene-Level Biological Intelligence</h2>
              <p className="text-slate-400 text-lg leading-relaxed max-w-3xl">
                How individual genes contribute to modeled survival risk in the Cox proportional hazards framework.
              </p>
            </div>
          </div>

          {/* PORTAL / CTA CARD */}
          <div className="bg-[#0f172a] rounded-3xl border border-white/10 p-10 md:p-16 text-center relative overflow-hidden group">
            
            {/* Ambient Background Glow */}
            <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-emerald-600/10 rounded-full blur-[100px] pointer-events-none group-hover:bg-emerald-600/20 transition-all duration-700"></div>
            
            <div className="relative z-10 max-w-2xl mx-auto">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 mb-8 group-hover:scale-110 transition-transform duration-300">
                <Icon name="Database" size={32} className="text-emerald-400" />
              </div>
              
              <h3 className="text-2xl md:text-3xl font-bold text-white mb-6">
                Explore the Full 31-Gene Signature
              </h3>
              
              <p className="text-slate-400 text-lg leading-relaxed mb-8">
                Access the dedicated biological intelligence engine to view detailed coefficients, 
                rankings, and narrative descriptions for every gene in the OncoRisk panel.
              </p>
              
              <Link to="/clinical-implementation">
                <Button 
                  variant="default" 
                  size="lg" 
                  className="bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-900/20"
                  iconName="ArrowRight"
                  iconPosition="right"
                >
                  Open Gene Narrative Engine
                </Button>
              </Link>

              <div className="mt-8 flex items-center justify-center gap-2 text-xs text-slate-500">
                <Icon name="ShieldCheck" size={14} />
                <span>Includes Immune, Metabolic, and ECM Program Analysis</span>
              </div>
            </div>
          </div>
        </div>

      </section>
    </div>
  );
};

export default AnalyticsContent;