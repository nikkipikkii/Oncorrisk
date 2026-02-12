import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';

const TechnicalDeepDive = () => {
  const [activeTab, setActiveTab] = useState('logic');

  return (
    <section className="py-24 bg-[#080c14] text-white border-t border-white/5 relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-blue-600/5 rounded-full blur-[100px] pointer-events-none"></div>
      
      <div className="container mx-auto px-4 md:px-6 lg:px-8 max-w-6xl relative z-10">
        
        {/* ==========================================
            01. EXTERNAL VALIDATION
           ========================================== */}
        <div className="mb-24">
          <div className="flex items-baseline gap-6 mb-12 border-b border-white/5 pb-8">
            <span className="text-6xl md:text-8xl font-black text-white/10 leading-none select-none">01</span>
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">External Validation & Generalization</h2>
              <p className="text-slate-400 text-lg leading-relaxed max-w-3xl">
                Model development and evaluation used two large, independent breast cancer cohorts that differ in origin, measurement technology, and clinical context. 
                This separation was intentional: one cohort was used exclusively for development, while the other was reserved strictly for <strong className="text-white">external validation.</strong>
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-8 md:gap-12">
            <div className="bg-slate-900/50 p-8 rounded-3xl border border-white/5 h-full">
              <h3 className="font-bold text-xl mb-4 flex items-center gap-3">
                <span className="w-2.5 h-2.5 bg-blue-500 rounded-full"></span>
                TCGA-BRCA: Training & Internal Eval
              </h3>
              <p className="text-slate-400 mb-8 leading-relaxed">
                Primary development cohort with 1,059 patients. A frozen train–test split was enforced throughout:
              </p>
              <ul className="grid grid-cols-2 gap-4 text-center">
                <li className="bg-black/20 p-4 rounded-2xl border border-white/5">
                  <div className="text-2xl font-bold text-white">847</div>
                  <div className="text-[10px] uppercase text-slate-500 font-bold mt-1">Training set</div>
                </li>
                <li className="bg-black/20 p-4 rounded-2xl border border-white/5">
                  <div className="text-2xl font-bold text-white">212</div>
                  <div className="text-[10px] uppercase text-slate-500 font-bold mt-1">Internal test</div>
                </li>
              </ul>
            </div>

            <div className="bg-slate-900/50 p-8 rounded-3xl border border-white/5 h-full relative overflow-hidden">
              <div className="absolute top-0 right-0 w-24 h-24 bg-green-500/10 rounded-bl-full pointer-events-none"></div>
              <h3 className="font-bold text-xl mb-4 flex items-center gap-3 text-green-400">
                <Icon name="ShieldCheck" size={20} />
                METABRIC: External Validation
              </h3>
              <p className="text-slate-300 mb-6 leading-relaxed">
                Independently collected cohort (1,903 patients). No METABRIC samples were used during training or tuning. Enforces cross-cohort compatibility and reduces overfitting.
              </p>
              <div className="p-4 bg-green-500/5 border border-green-500/10 rounded-xl">
                <p className="text-sm text-green-400/90 italic font-medium">
                  "Generalization is treated as a first-class constraint rather than a post hoc check."
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* ==========================================
            02. HOW RISK IS INTERPRETED
           ========================================== */}
        <div className="mb-24">
          <div className="flex items-baseline gap-6 mb-12 border-b border-white/5 pb-8">
            <span className="text-6xl md:text-8xl font-black text-white/10 leading-none select-none">02</span>
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Inference Logic & Interpretation</h2>
              <div className="text-[10px] font-bold bg-white/10 px-3 py-1 rounded text-slate-400 uppercase tracking-widest inline-block">
                Research Depth
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-12 border-l-2 border-white/5 pl-8">
            <div className="space-y-4">
              <p className="text-slate-300 text-lg leading-relaxed">
                Rather than collapsing outputs into a single estimate, OncoRisk presents results from both modeling paradigms side by side.
              </p>
              <p className="text-base text-slate-400">
                When both models produce a similar survival story, confidence increases. When they diverge, <strong className="text-white">uncertainty is explicitly surfaced</strong> rather than smoothed away.
              </p>
            </div>
            <div className="bg-blue-500/5 p-8 rounded-3xl border border-blue-500/10 text-base leading-relaxed italic text-blue-300">
              "No single estimate is treated as absolute truth. This design keeps inference aligned with research use, emphasizing structured interpretation over automated decision-making."
            </div>
          </div>
        </div>

        {/* ==========================================
            03. WHY THESE INPUTS WERE CHOSEN
           ========================================== */}
        <div className="mb-24">
          <div className="flex items-baseline gap-6 mb-12 border-b border-white/5 pb-8">
            <span className="text-6xl md:text-8xl font-black text-white/10 leading-none select-none">03</span>
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Why These Inputs Were Chosen</h2>
              <p className="text-slate-400 text-lg leading-relaxed max-w-3xl">
                 Integrating clinical anchors with a curated molecular backbone to ensure robust external performance.
              </p>
            </div>
          </div>

          <div className="bg-[#0f172a] rounded-2xl border border-emerald-500/20 shadow-2xl overflow-hidden mb-12">
            <div className="grid md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-emerald-500/10">
              <div className="p-8 group hover:bg-white/[0.01] transition-colors relative">
                <div className="text-emerald-500 mb-3 font-bold text-xs tracking-widest uppercase flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)]"></span>
                  Predictor 01
                </div>
                <h4 className="text-xl font-bold mb-4 text-white">Lymph-Node Involvement</h4>
                <p className="text-sm text-slate-400 leading-relaxed">
                  Lymph-node status is one of the strongest real-world predictors of prognosis in breast cancer because it captures whether tumor cells have escaped the primary site. In this project’s experiments, including lymph-node positivity consistently improved separation between low- and high-risk groups and strengthened external performance.
                </p>
                <div className="absolute bottom-0 left-0 h-1 w-0 group-hover:w-full bg-emerald-500/40 transition-all duration-500 ease-out"></div>
              </div>

              <div className="p-8 group hover:bg-white/[0.01] transition-colors relative">
                <div className="text-emerald-500 mb-3 font-bold text-xs tracking-widest uppercase flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)]"></span>
                  Predictor 02
                </div>
                <h4 className="text-xl font-bold mb-4 text-white">Age at Diagnosis</h4>
                <p className="text-sm text-slate-400 leading-relaxed">
                  Age captures a mixture of immune, hormonal, and treatment-related effects that shift baseline risk even when tumors appear similar histologically. Across TCGA and METABRIC, age showed a stable association with hazard and remained reliable under external validation.
                </p>
                <div className="absolute bottom-0 left-0 h-1 w-0 group-hover:w-full bg-emerald-500/40 transition-all duration-500 ease-out"></div>
              </div>

              <div className="p-8 group hover:bg-white/[0.01] transition-colors relative">
                <div className="text-emerald-500 mb-3 font-bold text-xs tracking-widest uppercase flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)]"></span>
                  Molecular Backbone
                </div>
                <h4 className="text-xl font-bold mb-4 text-white">The 31-Gene Panel</h4>
                <div className="text-sm text-slate-400 leading-relaxed space-y-4">
                  <p>
                    OncoRisk relies on a curated 31-gene panel contributing non-random, directionally consistent signal across models. Functionally, they cluster into:
                  </p>
                  <ul className="space-y-2 border-l border-emerald-500/20 pl-4">
                    <li>• <span className="text-emerald-400 font-medium">Immune:</span> CD24, TNFRSF14, CCL19.</li>
                    <li>• <span className="text-emerald-400 font-medium">Metabolic:</span> SLC16A2, SERPINA1, QPRT.</li>
                    <li>• <span className="text-emerald-400 font-medium">ECM:</span> SEMA3B, TFPI2.</li>
                  </ul>
                </div>
                <div className="absolute bottom-0 left-0 h-1 w-0 group-hover:w-full bg-emerald-500/40 transition-all duration-500 ease-out"></div>
              </div>
            </div>

            <div className="p-8 bg-emerald-500/[0.02] border-t border-emerald-500/10">
              <div className="flex flex-col md:flex-row gap-8">
                <div className="md:w-1/2 space-y-4">
                  <div className="flex items-center gap-3">
                    <div className="px-3 py-1 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-bold uppercase tracking-widest">Insights</div>
                    <h4 className="text-xl font-bold text-white">What the Model Learns</h4>
                  </div>
                  <p className="text-slate-400 text-sm leading-relaxed">Both models consistently surface three dominant survival programs:</p>
                  <ul className="space-y-2 text-slate-300 text-sm list-disc pl-5">
                    <li>immune regulation at the tumor–immune interface</li>
                    <li>metabolic stress tolerance and mitochondrial function</li>
                    <li>extracellular matrix remodeling associated with invasion</li>
                  </ul>
                </div>
                <div className="md:w-1/2 flex items-center">
                  <p className="text-slate-400 text-sm leading-relaxed border-l-2 border-emerald-500/20 pl-6 italic">
                    High-risk tumors show coordinated activation of these programs, while low-risk tumors lack a dominant driver profile. This suggests aggressive disease follows structured biological strategies rather than random noise.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ==========================================
            04. WHAT THE PROJECT ACHIEVED
           ========================================== */}
        <div className="mb-24">
          <div className="flex items-baseline gap-6 mb-12 border-b border-white/5 pb-8">
            <span className="text-6xl md:text-8xl font-black text-white/10 leading-none select-none">04</span>
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">What the Project Achieved</h2>
              <p className="text-slate-400 text-lg leading-relaxed max-w-3xl">
                 Translating clinicogenomic profiles into transparent survival estimates and exposing biological programs under uncertainty.
              </p>
            </div>
          </div>

          <div className="bg-white/5 rounded-[2.5rem] p-10 md:p-16 text-center border border-white/10">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12">
              <div className="p-4 rounded-2xl bg-black/20">
                <div className="text-4xl font-bold text-cyan-400">0.79</div>
                <div className="text-xs text-slate-500 font-bold uppercase mt-2 tracking-wider">Cox C-index</div>
              </div>
              <div className="p-4 rounded-2xl bg-black/20">
                <div className="text-4xl font-bold text-pink-500">0.73</div>
                <div className="text-xs text-slate-500 font-bold uppercase mt-2 tracking-wider">RSF C-index</div>
              </div>
              <div className="p-4 rounded-2xl bg-black/20">
                <div className="text-4xl font-bold text-white">31</div>
                <div className="text-xs text-slate-500 font-bold uppercase mt-2 tracking-wider">Gene Signature</div>
              </div>
              <div className="p-4 rounded-2xl bg-black/20">
                <div className="text-4xl font-bold text-slate-400">0.63</div>
                <div className="text-xs text-slate-500 font-bold uppercase mt-2 tracking-wider">Ext. Validated</div>
              </div>
            </div>
            <div className="inline-flex items-center gap-3 px-6 py-3 bg-white/5 rounded-full border border-white/10 text-sm text-slate-400 font-medium">
              <Icon name="Shield" size={16} className="text-orange-500" />
              Not intended for clinical use • Research & Translational Project
            </div>
          </div>
        </div>

        {/* ==========================================
            05. RISK INFERENCE LOGIC (Project Synthesis)
           ========================================== */}
        <div>
          <div className="flex items-baseline gap-6 mb-12 border-b border-white/5 pb-8">
            <span className="text-6xl md:text-8xl font-black text-white/10 leading-none select-none">05</span>
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-bold uppercase tracking-widest mb-4">
                <Icon name="FileText" size={14} />
                <span>Project Synthesis</span>
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Risk Inference Logic & Architecture</h2>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-12">
            <div className="bg-slate-900/50 border border-white/10 rounded-3xl p-10 flex flex-col justify-center h-full relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/10 rounded-bl-full blur-2xl"></div>
              <h3 className="text-2xl font-bold text-white mb-6">Twin-Model Inference</h3>
              <p className="text-slate-300 leading-relaxed mb-8 text-lg">
                OncoRisk avoids collapsing outputs into a single estimate. <strong className="text-cyan-400">Agreement builds confidence, while divergence surfaces uncertainty.</strong>
              </p>
              <div className="bg-black/20 rounded-2xl p-6 border border-white/5">
                <div className="flex items-center gap-3 mb-3">
                  <Icon name="Shield" size={18} className="text-cyan-400" />
                  <span className="font-semibold text-white text-sm uppercase tracking-wide">Research-First Design</span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                  No single estimate is treated as absolute truth. This keeps inference aligned with translational research rather than automated decision-making.
                </p>
              </div>
            </div>

            <div className="space-y-6">
              <div className="flex gap-6 bg-slate-900/30 p-6 rounded-2xl border border-white/5 group">
                <div className="h-12 w-12 rounded-full bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center text-cyan-400 font-bold text-lg flex-shrink-0 group-hover:bg-cyan-500 group-hover:text-white transition-all">1</div>
                <div>
                  <h4 className="font-bold text-white text-lg mb-1">Independent Computation</h4>
                  <p className="text-sm text-slate-400">Cox and RSF compute survival curves and RMST separately using distinct mathematical paradigms.</p>
                </div>
              </div>
              
              <div className="flex gap-6 bg-slate-900/30 p-6 rounded-2xl border border-white/5 group">
                <div className="h-12 w-12 rounded-full bg-purple-500/20 border border-purple-500/30 flex items-center justify-center text-purple-400 font-bold text-lg flex-shrink-0 group-hover:bg-purple-500 group-hover:text-white transition-all">2</div>
                <div>
                  <h4 className="font-bold text-white text-lg mb-1">Agreement Analysis</h4>
                  <p className="text-sm text-slate-400">A logic score quantifies alignment. Convergent results indicate High Confidence in the prediction.</p>
                </div>
              </div>

              <div className="flex gap-6 bg-slate-900/30 p-6 rounded-2xl border border-white/5 group">
                <div className="h-12 w-12 rounded-full bg-green-500/20 border border-green-500/30 flex items-center justify-center text-green-400 font-bold text-lg flex-shrink-0 group-hover:bg-green-500 group-hover:text-white transition-all">3</div>
                <div>
                  <h4 className="font-bold text-white text-lg mb-1">Structured Output</h4>
                  <p className="text-sm text-slate-400">Results are presented for researcher interpretation, allowing for nuanced assessment of disease trajectories.</p>
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </section>
  );
};

export default TechnicalDeepDive;