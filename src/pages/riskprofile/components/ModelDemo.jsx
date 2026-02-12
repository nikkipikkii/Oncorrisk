import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { Activity, GitCommit, AlertCircle, Clock, ChevronDown, ChevronUp, Play, RotateCcw, Info } from 'lucide-react';

const ModelDemo = () => {
  // --- STATE MANAGEMENT ---
  const [mode, setMode] = useState('manual');
  const [selectedPid, setSelectedPid] = useState('');
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenesOpen, setIsGenesOpen] = useState(false);
  
  // Data from Backend
  const [existingPatients, setExistingPatients] = useState([]);
  const [geneList, setGeneList] = useState([]);

  // Form State
  const [formData, setFormData] = useState({
    age: '',
    nodeStatus: 'Negative',
    genes: {} 
  });

  // --- INITIAL LOAD: FETCH PATIENTS & GENES FROM BACKEND ---
  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/metadata');
        if (!response.ok) throw new Error('Failed to fetch metadata');
        const data = await response.json();
        
        setExistingPatients(data.patients);
        setGeneList(data.genes);
        
        // Initialize gene form state with 0.0 based on actual model features
        const initialGenes = data.genes.reduce((acc, gene) => ({ ...acc, [gene]: 0.0 }), {});
        setFormData(prev => ({ ...prev, genes: initialGenes }));
      } catch (err) {
        console.error("Metadata fetch error:", err);
      }
    };
    fetchMetadata();
  }, []);

  // --- HANDLERS ---
  const handleModeChange = (newMode) => {
    setMode(newMode);
    setResults(null);
    if (newMode === 'manual') {
        const resetGenes = geneList.reduce((acc, gene) => ({ ...acc, [gene]: 0.0 }), {});
        setFormData({ age: '', nodeStatus: 'Negative', genes: resetGenes });
    }
  };

  const handlePatientSelect = (e) => {
    const pid = e.target.value;
    setSelectedPid(pid);
    const patient = existingPatients.find(p => p.id === pid);
    if (patient) {
      setFormData(prev => ({ 
        ...prev, 
        age: patient.age, 
        nodeStatus: patient.node === 1 ? 'Positive' : 'Negative' 
      }));
    }
  };

  const handleGeneChange = (geneKey, value) => {
    setFormData(prev => ({
      ...prev,
      genes: { ...prev.genes, [geneKey]: parseFloat(value) || 0 }
    }));
  };

  const resetForm = () => {
    setMode('manual');
    setSelectedPid('');
    setResults(null);
    setIsGenesOpen(false);
    const resetGenes = geneList.reduce((acc, gene) => ({ ...acc, [gene]: 0.0 }), {});
    setFormData({
      age: '',
      nodeStatus: 'Negative',
      genes: resetGenes
    });
  };

  // --- ACTUAL BACKEND INFERENCE CALL ---
  const runInference = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          age: parseFloat(formData.age),
          nodeStatus: formData.nodeStatus,
          genes: formData.genes
        }),
      });

      if (!response.ok) throw new Error('Inference request failed');
      
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Inference Error:", error);
      alert("Backend Connection Error: Ensure main.py is running on port 8000");
    } finally {
      setIsLoading(false);
    }
  };

  // Common Styles
  const inputClass = "w-full bg-[#0f172a] border border-slate-700 rounded-lg p-3 text-slate-200 focus:outline-none focus:border-pink-500 transition-colors";
  const labelClass = "block text-sm font-medium text-slate-400 mb-2";
  const cardClass = "bg-[#1e293b] rounded-xl border border-slate-700/50 p-6";

  return (
    <div className="min-h-screen bg-[#0B1120] text-slate-200 font-sans p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        
        {/* HEADER */}
        <div className="flex justify-between items-start mb-8 border-b border-slate-800 pb-6">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <Activity className="w-6 h-6 text-pink-500" />
              <h1 className="text-2xl font-bold text-white tracking-tight">OncoRisk Research Model</h1>
            </div>
            <p className="text-slate-500 text-sm">Cox + RSF Dual Inference System</p>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isLoading ? 'bg-amber-500' : 'bg-emerald-500'} animate-pulse`}></div>
            <span className={`${isLoading ? 'text-amber-500' : 'text-emerald-500'} text-xs font-medium uppercase tracking-wider`}>
                {isLoading ? 'Processing' : 'System Ready'}
            </span>
          </div>
        </div>

        {/* --- INPUT CARD --- */}
        <div className="bg-[#1e293b] rounded-xl border border-slate-700/50 shadow-xl overflow-hidden mb-8">
          
          {/* TABS */}
          <div className="grid grid-cols-2 p-6 gap-4 border-b border-slate-700/50">
            <button
              onClick={() => handleModeChange('manual')}
              className={`flex items-center justify-center gap-2 py-3 rounded-lg border font-medium transition-all ${
                mode === 'manual'
                  ? 'border-pink-500 text-pink-500 bg-pink-500/5 shadow-[0_0_15px_rgba(236,72,153,0.1)]'
                  : 'border-slate-700 text-slate-400 hover:bg-slate-800'
              }`}
            >
              <span>✎ Manual Profile</span>
            </button>
            <button
              onClick={() => handleModeChange('existing')}
              className={`flex items-center justify-center gap-2 py-3 rounded-lg border font-medium transition-all ${
                mode === 'existing'
                  ? 'border-blue-500 text-blue-400 bg-blue-500/5'
                  : 'border-slate-700 text-slate-400 hover:bg-slate-800'
              }`}
            >
              <span>🗄 Select TCGA Patient</span>
            </button>
          </div>

          <div className="p-6 md:p-8 space-y-8">
            {/* CLINICAL INPUTS */}
            <div>
              <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                <Activity className="w-4 h-4 text-blue-400" /> Clinical Inputs
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  {mode === 'existing' ? (
                    <>
                      <label className={labelClass}>Select TCGA Patient ID</label>
                      <select className={inputClass} value={selectedPid} onChange={handlePatientSelect}>
                        <option value="">-- Select Patient --</option>
                        {existingPatients.map(p => <option key={p.id} value={p.id}>{p.id}</option>)}
                      </select>
                    </>
                  ) : (
                    <>
                      <label className={labelClass}>Age (years)</label>
                      <input type="number" placeholder="Enter patient age" className={inputClass}
                        value={formData.age} onChange={(e) => setFormData({...formData, age: e.target.value})}
                      />
                      <p className="mt-2 text-xs text-slate-500">Age at diagnosis (baseline modifier)</p>
                    </>
                  )}
                </div>
                <div>
                  <label className={labelClass}>Lymph Node Status</label>
                  <select disabled={mode === 'existing'}
                    className={`${inputClass} ${mode === 'existing' ? 'opacity-50 cursor-not-allowed' : ''}`}
                    value={formData.nodeStatus} onChange={(e) => setFormData({...formData, nodeStatus: e.target.value})}
                  >
                    <option value="Negative">Negative</option>
                    <option value="Positive">Positive</option>
                  </select>
                  <p className="mt-2 text-xs text-slate-500">NODE_POS indicator</p>
                </div>
              </div>
            </div>

            {/* MOLECULAR PROFILE (DROPDOWN) */}
            <div className="border-t border-slate-700 pt-6">
              <button 
                onClick={() => setIsGenesOpen(!isGenesOpen)}
                className="w-full flex items-center justify-between bg-[#0f172a] p-4 rounded-lg border border-slate-700 hover:border-pink-500/50 transition-colors group"
              >
                <div className="flex items-center gap-3">
                    <GitCommit className="w-5 h-5 text-pink-500" />
                    <div className="text-left">
                        <span className="block text-white font-medium group-hover:text-pink-400 transition-colors">Molecular Profile (Optional)</span>
                        <span className="text-xs text-slate-500">Configure {geneList.length}-gene expression panel (z-scored)</span>
                    </div>
                </div>
                {isGenesOpen ? <ChevronUp className="text-slate-400"/> : <ChevronDown className="text-slate-400"/>}
              </button>

              {isGenesOpen && (
                <div className="mt-4 bg-[#0f172a] rounded-xl border border-slate-700 p-4 animate-fade-in">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                    {geneList.map((gene) => (
                      <div key={gene}>
                        <label className="block text-xs font-medium text-slate-400 mb-1.5">{gene}</label>
                        <input
                          type="number" step="0.1" placeholder="0.0"
                          className="w-full bg-[#1e293b] border border-slate-700 rounded p-2 text-sm text-white focus:border-pink-500 focus:outline-none placeholder-slate-600"
                          value={formData.genes[gene] || 0.0}
                          onChange={(e) => handleGeneChange(gene, e.target.value)}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* ACTIONS */}
            <div className="flex gap-4 pt-4 border-t border-slate-800">
              <button
                onClick={runInference}
                disabled={isLoading || (mode === 'existing' && !selectedPid) || !formData.age}
                className={`flex-1 py-4 rounded-lg font-semibold text-white shadow-lg shadow-blue-900/20 flex items-center justify-center gap-2 transition-all ${
                  isLoading ? 'bg-slate-700 cursor-wait' : 'bg-blue-600 hover:bg-blue-500'
                }`}
              >
                {isLoading ? <span className="animate-pulse">Running Analysis...</span> : <><Play className="w-4 h-4 fill-current" /> Run Dual-Model Analysis</>}
              </button>
              
              <button onClick={resetForm} className="px-6 py-4 rounded-lg border border-slate-600 text-slate-300 font-medium hover:bg-slate-800 transition-colors">
                <RotateCcw className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* --- RESULTS SECTION --- */}
        {results && (
          <div className="space-y-8 animate-fade-in-up pb-12">
            
            {/* 1. RISK SUMMARY (HIGH LEVEL) */}
            <section>
                <div className="flex items-center gap-2 mb-4">
                     <AlertCircle className="w-5 h-5 text-pink-500" />
                     <h2 className="text-xl font-bold text-white">Risk Summary (High-Level)</h2>
                </div>
               
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className={cardClass}>
                        <h4 className="text-slate-200 font-bold uppercase tracking-wider mb-2">Relative Risk (Cox Hazard)</h4>
                        <p className="text-slate-500 text-xs mb-4 min-h-[32px]">Measures risk relative to an average patient in the training cohort.</p>
                        <div className="text-4xl font-bold text-blue-400">{results.summary.coxHazard}</div>
                    </div>

                    <div className={cardClass}>
                        <h4 className="text-slate-200 font-bold uppercase tracking-wider mb-2">RSF Risk (cum. hazard)</h4>
                        <p className="text-slate-500 text-xs mb-4 min-h-[32px]">Captures nonlinear risk accumulation over time.</p>
                        <div className="text-4xl font-bold text-emerald-400">{results.summary.rsfRisk}</div>
                    </div>

                    <div className={cardClass}>
                       <h4 className="text-slate-200 font-bold uppercase tracking-wider mb-2">Model Agreement</h4>
                        <p className="text-slate-500 text-xs mb-4 min-h-[32px]">Indicates whether two different modeling assumptions tell a similar survival story.</p>
                        <div className="flex items-baseline gap-2">
                             <span className="text-4xl font-bold text-purple-400">{results.summary.agreementLabel}</span>
                             <span className="text-sm text-slate-500">({results.summary.agreement})</span>
                        </div>
                    </div>
                </div>

                <div className="mt-4 p-4 bg-blue-900/20 border border-blue-500/20 rounded-lg flex gap-3">
                    <Info className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
                    <p className="text-sm text-blue-200">
                        <strong className="text-blue-400">Interpretation Hint:</strong> Lower values indicate lower relative risk compared to the training population, 
                        not zero risk. This prevents catastrophic misreading.
                    </p>
                </div>
            </section>

            {/* 2. SURVIVAL TIME ESTIMATES */}
            <section>
              <div className="flex items-center gap-2 mb-4">
                <Clock className="w-5 h-5 text-blue-400" />
                <h2 className="text-xl font-bold text-white">Survival Time Estimates (model-based)</h2>
              </div>

              <div className="mb-8 text-sm text-slate-400">
                <p className="mb-2">
                  Time measured from diagnosis or study start until an event in the dataset. These are statistical summaries of modeled survival curves,
                  not predictions of how long an individual will live.
                </p>
                <ul className="list-disc list-inside text-xs text-slate-500 space-y-1 ml-2">
                  <li>Values summarize expected survival behavior across similar patients, not a fixed timeline.</li>
                  <li>"Not reached" means the modeled survival probability never dropped below 50% within the observed follow-up window.</li>
                </ul>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="bg-[#0f172a] rounded-lg p-5 border border-slate-800">
                  <h4 className="text-blue-400 font-bold text-lg mb-1">Cox Median Survival</h4>
                  <div className="text-3xl font-bold text-white mb-3">
                    {results.estimates.medianCox ? `${results.estimates.medianCox} days` : "Not reached"}
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    Median survival time inferred from the Cox survival curve. Reported as “Not reached” if survival probability does not fall below 50% within the observed follow-up window.
                  </p>
                </div>

                <div className="bg-[#0f172a] rounded-lg p-5 border border-slate-800">
                  <h4 className="text-emerald-400 font-bold text-lg mb-1">RSF Median Survival</h4>
                  <div className="text-3xl font-bold text-white mb-3">
                    {results.estimates.medianRsf ? `${results.estimates.medianRsf} days` : "Not reached"}
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    Median survival time inferred from the Random Survival Forest survival curve. Represents the time at which modeled survival probability falls below 50%.
                  </p>
                </div>

                <div className="bg-[#0f172a] rounded-lg p-5 border border-slate-800">
                  <h4 className="text-purple-400 font-bold text-lg mb-1">Consensus Median</h4>
                  <div className="text-3xl font-bold text-white mb-3">
                    {results.estimates.consensus ? `${results.estimates.consensus} days` : "Not reached"}
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    Median survival estimate when both models report a similar value. Shown for comparison only; no single value is treated as absolute truth.
                  </p>
                </div>
              </div>
            </section>
            
            {/* 3. RMST */}
            <section>
                 <h2 className="text-xl font-bold text-white mb-4">Restricted Mean Survival Time (RMST)</h2>
                 <p className="text-slate-400 text-sm mb-6">
                    RMST summarizes the average survival time within the observed follow-up period, even when median survival is not reached.
                 </p>

                 <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className={cardClass}>
                        <div className="flex justify-between items-start mb-3">
                            <div>
                                <h4 className="text-slate-200 font-bold">Cox RMST</h4>
                                <div className="text-xs text-slate-500 mt-1 max-w-xs">
                                    Restricted Mean Survival Time estimated from the Cox model. Represents the average survival time within the observed follow-up period.
                                </div>
                            </div>
                            <span className="text-2xl font-bold text-blue-400 font-mono">{results.rmst.cox} days</span>
                        </div>
                    </div>

                    <div className={cardClass}>
                        <div className="flex justify-between items-start mb-3">
                            <div>
                                <h4 className="text-slate-200 font-bold">RSF RMST</h4>
                                <div className="text-xs text-slate-500 mt-1 max-w-xs">
                                    Restricted Mean Survival Time estimated from the Random Survival Forest. Useful for comparing expected survival when curves do not cross the 50% threshold.
                                </div>
                            </div>
                            <span className="text-2xl font-bold text-emerald-400 font-mono">{results.rmst.rsf} days</span>
                        </div>
                    </div>
                 </div>
            </section>

             {/* 4. SURVIVAL CURVES (GRAPH) */}
             {/* <section className="bg-[#1e293b] rounded-xl border border-slate-700/50 p-6 h-[400px]">
                <h3 className="text-white font-bold mb-4">Survival Curves</h3>
                <ResponsiveContainer width="100%" height="90%">
                <LineChart data={results.curveData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
                    <XAxis dataKey="time" stroke="#94a3b8" tick={{fill: '#94a3b8', fontSize: 12}} />
                    <YAxis stroke="#94a3b8" tick={{fill: '#94a3b8', fontSize: 12}} domain={[0, 1.05]} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#e2e8f0' }} />
                    <Legend verticalAlign="top" height={36}/>
                    <Line type="monotone" dataKey="cox" stroke="#3b82f6" name="CoxPH Model" strokeWidth={3} dot={false} />
                    <Line type="monotone" dataKey="rsf" stroke="#10b981" name="RSF Model" strokeWidth={3} strokeDasharray="4 4" dot={false} />
                    {results.estimates.medianRsf && (
                        <ReferenceLine x={results.estimates.medianRsf} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'insideTopRight', value: 'Median (RSF)', fill: '#ef4444', fontSize: 12 }} />
                    )}
                </LineChart>
                </ResponsiveContainer>
            </section> */}

            {/* 4. SURVIVAL CURVES (GRAPH) */}
<section className="bg-[#1e293b] rounded-xl border border-slate-700/50 p-6 h-[450px]">
  <h3 className="text-white font-bold mb-4">Survival Curves</h3>
  <ResponsiveContainer width="100%" height="90%">
    <LineChart
      data={results.curveData}
      margin={{ top: 20, right: 30, left: 20, bottom: 40 }}
    >
      <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
      
      <XAxis
        dataKey="time"
        stroke="#94a3b8"
        tick={{ fill: '#94a3b8', fontSize: 12 }}
        // Set fixed intervals: 0, 2000, 4000, 6000, etc.
        ticks={[0, 2000, 4000, 6000, 8000]}
        type="number"
        domain={[0, 'dataMax']}
        label={{ 
          value: 'Time (days)', 
          position: 'insideBottom', 
          offset: -25, 
          fill: '#94a3b8',
          fontSize: 14,
          fontWeight: 'bold'
        }}
      />
      
      <YAxis
        stroke="#94a3b8"
        tick={{ fill: '#94a3b8', fontSize: 12 }}
        domain={[0, 1.05]}
        label={{ 
          value: 'Survival Probability', 
          angle: -90, 
          position: 'insideLeft', 
          offset: 0, 
          fill: '#94a3b8',
          fontSize: 14,
          fontWeight: 'bold'
        }}
      />
      
      <Tooltip
        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
        itemStyle={{ fontSize: '12px' }}
        labelStyle={{ color: '#94a3b8', marginBottom: '4px' }}
        // Format the tooltip label to show "Day: X"
        labelFormatter={(value) => `Day: ${Math.round(value)}`}
      />
      
      <Legend verticalAlign="top" height={40} />
      
      <Line
        type="monotone"
        dataKey="cox"
        stroke="#3b82f6"
        name="CoxPH Model"
        strokeWidth={3}
        dot={false}
        // Shows point on hover
        activeDot={{ r: 6, stroke: '#3b82f6', strokeWidth: 2, fill: '#0f172a' }}
      />
      
      <Line
        type="monotone"
        dataKey="rsf"
        stroke="#10b981"
        name="RSF Model"
        strokeWidth={3}
        strokeDasharray="4 4"
        dot={false}
        // Shows point on hover
        activeDot={{ r: 6, stroke: '#10b981', strokeWidth: 2, fill: '#0f172a' }}
      />
      
      {/* Vertical Median Line */}
      {results.estimates.medianRsf && (
        <ReferenceLine
          x={results.estimates.medianRsf}
          stroke="#ef4444"
          strokeDasharray="3 3"
          label={{
            position: 'insideTopRight',
            value: `Median: ${Math.round(results.estimates.medianRsf)}d`,
            fill: '#ef4444',
            fontSize: 12,
            fontWeight: 'bold',
            backgroundColor: 'rgba(15, 23, 42, 0.8)'
          }}
        />
      )}
    </LineChart>
  </ResponsiveContainer>
</section>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelDemo;