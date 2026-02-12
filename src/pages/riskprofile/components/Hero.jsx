import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import dnaImage from './dna_nobg.png';

const Hero = () => {
  const [openItem, setOpenItem] = useState(null);

  const guideItems = [
    {
      id: 1,
      title: "Input Mode",
      content: (
        <div className="space-y-3">
          <p>You can evaluate risk in two ways:</p>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong className="text-white">Manual profile:</strong> enter a hypothetical or approximate patient profile.</li>
            <li><strong className="text-white">Select existing TCGA test patient:</strong> load a real, held-out patient from the dataset.</li>
          </ul>
          <p className="text-sm italic text-slate-400">This distinction exists to separate exploration from grounded reference cases.</p>
        </div>
      )
    },
    {
      id: 2,
      title: "Clinical Inputs",
      content: (
        <div className="space-y-4">
          <div>
            <h4 className="font-bold text-pink-500 mb-1">Age (years)</h4>
            <p>Age is treated as a baseline risk modifier. Higher age shifts risk upward on average, but does not determine outcome on its own.</p>
          </div>
          <div>
            <h4 className="font-bold text-pink-500 mb-1">Lymph Node Status</h4>
            <p>Captures whether cancer cells have spread to nearby lymph nodes. Negative indicates lower baseline risk, while Positive indicates higher risk. This is one of the most stable predictors across cohorts.</p>
          </div>
        </div>
      )
    },
    {
      id: 3,
      title: "Molecular Profile (Optional)",
      content: (
        <div className="space-y-3">
          <p className="font-bold text-cyan-400 uppercase text-xs tracking-widest">What these gene values mean</p>
          <p>Each gene input represents <span className="text-white">relative expression</span>, not raw lab values. 0 is average, positive is higher-than-average, and negative is lower-than-average.</p>
          <div className="bg-orange-500/10 border border-orange-500/20 p-3 rounded-lg text-xs text-orange-200">
            <p><strong>Note:</strong> This section is intended for researchers and simulated experiments. Leaving all values at 0 isolates clinical risk by assuming an average molecular profile.</p>
          </div>
        </div>
      )
    },
    {
      id: 4,
      title: "Risk Scores (Model Outputs)",
      content: (
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-slate-800/40 p-3 rounded-lg border border-white/5">
            <h5 className="font-bold text-blue-400 text-sm mb-1">Cox Hazard (relative)</h5>
            <p className="text-xs text-slate-300">Answers: "Compared to other patients, is this profile higher or lower risk?"</p>
          </div>
          <div className="bg-slate-800/40 p-3 rounded-lg border border-white/5">
            <h5 className="font-bold text-pink-400 text-sm mb-1">RSF Risk (cumulative)</h5>
            <p className="text-xs text-slate-300">Reflects accumulated risk over time using a nonlinear model to capture interactions.</p>
          </div>
          <p className="col-span-2 text-xs text-slate-400 italic">Agreement between models builds confidence; divergence is surfaced as uncertainty.</p>
        </div>
      )
    },
    {
      id: 5,
      title: "Survival Time Estimates",
      content: (
        <div className="space-y-3">
          <p className="text-white font-bold">Days do NOT mean "how long the person will live."</p>
          <p>They indicate statistical summaries of time measured from diagnosis until a dataset event. "Not reached" means survival probability never dropped below 50% during follow-up.</p>
        </div>
      )
    },
    {
      id: 6,
      title: "Restricted Mean Survival Time (RMST)",
      content: (
        <div className="space-y-2">
          <p className="font-bold text-white">The average amount of survival time within a fixed window.</p>
          <p>Think of it as the area under the survival curve. It is often preferred in research because it is more stable than the median and does not require survival to reach 50%.</p>
        </div>
      )
    },
    {
      id: 7,
      title: "Survival Curves (Cox vs RSF)",
      content: (
        <div className="space-y-2">
          <p>This plot shows survival probability over time. Shape differences reveal variations in modeling assumptions, while alignment indicates a stable survival signal.</p>
          <div className="flex gap-4 text-xs font-bold uppercase pt-2">
            <span className="text-blue-500 underline underline-offset-4">Cox (Blue)</span>
            <span className="text-pink-500 border-b border-pink-500/50">RSF (Orange Dashed)</span>
          </div>
        </div>
      )
    }
  ];

  return (
    <section className="relative bg-[#0b1120] min-h-screen pt-32 pb-20 overflow-hidden">
      
      {/* 3D DNA OVERLAY (Corner Breaking Effect) */}
      {/* <div className="absolute top-[-5%] left-[-8%] w-[500px] md:w-[700px] pointer-events-none z-30 opacity-70 mix-blend-screen">
          <img 
            src={dnaImage} 
            alt="DNA Structure" 
            className="w-full h-auto transform -rotate-12 animate-pulse-slow" 
          />
      </div> */}

      <div className="container mx-auto px-6 md:px-12 lg:pl-48 relative z-20">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-start">
          
          {/* LEFT: Title & Explainer */}
          <div className="space-y-8">
            <div className="space-y-2">
              <h1 className="text-5xl md:text-7xl font-bold tracking-tighter text-white">
                Patient Risk <br/>
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-pink-500 to-cyan-400">Profile Inference</span>
              </h1>
              <p className="text-cyan-400 font-bold tracking-[0.3em] text-xs uppercase opacity-80">
                Cox Proportional Hazards + Random Survival Forest
              </p>
            </div>

            {/* TOP EXPLAINER BOX (Visible Box Template) */}
            <div className="bg-slate-900/60 backdrop-blur-md border border-white/10 rounded-2xl p-8 relative overflow-hidden group">
              <div className="absolute top-0 left-0 w-1.5 h-full bg-pink-500"></div>
              <h3 className="text-white font-bold text-xl mb-4 flex items-center gap-2">
                <Icon name="Info" size={20} className="text-pink-500" />
                What this page is
              </h3>
              <div className="text-slate-300 leading-relaxed space-y-4">
                <p>
                  This page explores how the trained survival models interpret risk for a single patient-like profile. 
                  <strong className="text-white"> It does not predict how long someone will live and is not intended for clinical use.</strong>
                </p>
                <ul className="space-y-2">
                  {[
                    "How different models assess relative risk",
                    "How survival probability evolves over time",
                    "Where models agree and where uncertainty appears"
                  ].map((text, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm">
                      <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-cyan-400"></div>
                      {text}
                    </li>
                  ))}
                </ul>
                <p className="text-xs italic pt-4 border-t border-white/5 opacity-60">
                  All outputs are comparative and probabilistic, not deterministic.
                </p>
              </div>
            </div>
          </div>

          {/* RIGHT: Numbered FAQ Accordion (Templates) */}
          <div className="bg-slate-900/40 border border-white/10 rounded-3xl overflow-hidden backdrop-blur-sm">
            {guideItems.map((item, idx) => (
              <div key={item.id} className="relative">
                <button 
                  onClick={() => setOpenItem(openItem === item.id ? null : item.id)}
                  className="w-full flex items-center justify-between p-6 transition-all hover:bg-white/5"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-pink-500 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-pink-500/20">
                      {item.id}
                    </div>
                    <span className={`font-bold tracking-tight transition-colors ${openItem === item.id ? 'text-white' : 'text-slate-400'}`}>
                      {item.title}
                    </span>
                  </div>
                  <Icon 
                    name="ChevronDown" 
                    size={18} 
                    className={`text-slate-500 transition-transform duration-300 ${openItem === item.id ? 'rotate-180' : ''}`} 
                  />
                </button>
                
                {/* Content Area (No separating lines as requested) */}
                <div 
                  className={`overflow-hidden transition-all duration-500 ease-in-out ${openItem === item.id ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0'}`}
                >
                  <div className="px-20 pb-8 text-sm text-slate-300 leading-relaxed">
                    {item.content}
                  </div>
                </div>
              </div>
            ))}
          </div>

        </div>
      </div>

      {/* SCROLL INDICATOR */}
      <div className="absolute bottom-10 left-0 right-0 flex flex-col items-center gap-2 animate-bounce opacity-40">
        <span className="text-[10px] text-white tracking-[0.2em] uppercase">Scroll to Interface</span>
        <Icon name="ArrowDown" size={16} className="text-cyan-400" />
      </div>

    </section>
  );
};

export default Hero;