import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  ScatterChart, Scatter, ZAxis, CartesianGrid, Legend 
} from 'recharts';
import { X, ChevronLeft, ChevronRight, Maximize2 } from 'lucide-react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

// --- MODAL COMPONENT ---
const CohortModal = ({ isOpen, onClose, initialIndex, chartData }) => {
  const [currentIndex, setCurrentIndex] = useState(initialIndex);

  // Sync index when opening
  useEffect(() => {
    if (isOpen) setCurrentIndex(initialIndex);
  }, [isOpen, initialIndex]);

  if (!isOpen) return null;

  const slides = [
    {
      id: 1,
      title: "TGCA Test-KM by Cox Median Hazard Split",
      description: "Kaplan-Meier survival curves stratified by the median Cox hazard score. The clear separation between the High Risk (Orange) and Low Risk (Blue) groups demonstrates the model's ability to stratify patients effectively within the internal test set.",
      type: "km",
      dataKey: "tcga_km"
    },
    {
      id: 2,
      title: "Metabric-KM by Cox Median Hazard Split",
      description: "External validation on the METABRIC cohort (n=1903). Even on this completely independent dataset, the model maintains significant stratification, proving that the biological signals captured are robust and not overfitted to TCGA.",
      type: "km",
      dataKey: "metabric_km"
    },
    {
      id: 3,
      title: "Cox vs RSF Agreement-TCGA Test",
      description: "A scatter plot comparing the linear risk score (Cox) against the nonlinear risk score (RSF). The strong diagonal alignment indicates that both models generally agree, but outliers reveal patients where nonlinear interactions drive risk differently.",
      type: "scatter",
      dataKey: "scatter"
    }
  ];

  const currentSlide = slides[currentIndex];
  const currentData = chartData ? chartData[currentSlide.dataKey] : [];

  const handlePrev = (e) => { e.stopPropagation(); setCurrentIndex((prev) => (prev === 0 ? slides.length - 1 : prev - 1)); };
  const handleNext = (e) => { e.stopPropagation(); setCurrentIndex((prev) => (prev === slides.length - 1 ? 0 : prev + 1)); };

  // Render Full Detail Chart with Labels
  const renderDetailChart = () => {
    if (!currentData || currentData.length === 0) return <div className="text-white/50 h-full flex items-center justify-center">Loading Data...</div>;

    if (currentSlide.type === "km") {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
            <XAxis 
              dataKey="time" 
              stroke="#94a3b8" 
              tick={{ fontSize: 12, fill: '#94a3b8' }}
              label={{ value: 'Time (days)', position: 'insideBottom', offset: -25, fill: '#cbd5e1', fontSize: 14 }} 
            />
            <YAxis 
              stroke="#94a3b8" 
              tick={{ fontSize: 12, fill: '#94a3b8' }} 
              domain={[0, 1]}
              label={{ value: 'Survival Probability', angle: -90, position: 'insideLeft', offset: 0, fill: '#cbd5e1', fontSize: 14 }}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f1f5f9' }}
              labelFormatter={(v) => `Time: ${v}`}
            />
            <Legend verticalAlign="top" height={36} />
            <Line type="stepAfter" data={currentData} dataKey="Low" stroke="#3b82f6" strokeWidth={3} dot={false} name="Low Risk" />
            <Line type="stepAfter" data={currentData} dataKey="High" stroke="#ec4899" strokeWidth={3} dot={false} name="High Risk" />
          </LineChart>
        </ResponsiveContainer>
      );
    } else {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
            <XAxis 
              type="number" 
              dataKey="x" 
              name="Cox Hazard" 
              stroke="#94a3b8"
              label={{ value: 'CoxPH Hazard', position: 'insideBottom', offset: -25, fill: '#cbd5e1', fontSize: 14 }}
            />
            <YAxis 
              type="number" 
              dataKey="y" 
              name="RSF Risk" 
              stroke="#94a3b8"
              label={{ value: 'RSF Risk (Cum. Hazard)', angle: -90, position: 'insideLeft', offset: 0, fill: '#cbd5e1', fontSize: 14 }}
            />
            <ZAxis range={[60, 60]} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f1f5f9' }} />
            <Scatter name="Patients" data={currentData} fill="#10b981" fillOpacity={0.6} />
          </ScatterChart>
        </ResponsiveContainer>
      );
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200" onClick={onClose}>
      {/* Modal Content - Click propagation stopped so clicking inside doesn't close it */}
      <div 
        className="relative w-full max-w-6xl h-[85vh] bg-gradient-to-br from-[#0B1120] via-[#1e293b]/90 to-[#0B1120] rounded-2xl border border-white/10 shadow-2xl overflow-hidden flex flex-col md:flex-row backdrop-blur-md"
        onClick={(e) => e.stopPropagation()}
      >
        <button 
          onClick={onClose}
          className="absolute top-4 right-4 z-50 p-2 bg-white/5 hover:bg-white/10 rounded-full text-white/70 hover:text-white transition-colors"
        >
          <X size={24} />
        </button>

        {/* Left: Graph Area */}
        <div className="w-full md:w-2/3 h-1/2 md:h-full p-6 flex flex-col relative group bg-black/20">
           {/* Navigation Arrows */}
           <button onClick={handlePrev} className="absolute left-4 top-1/2 -translate-y-1/2 p-3 bg-black/40 hover:bg-black/60 text-white rounded-full transition-all opacity-0 group-hover:opacity-100 z-10">
              <ChevronLeft size={24} />
           </button>
           <button onClick={handleNext} className="absolute right-4 top-1/2 -translate-y-1/2 p-3 bg-black/40 hover:bg-black/60 text-white rounded-full transition-all opacity-0 group-hover:opacity-100 z-10">
              <ChevronRight size={24} />
           </button>
           
           <div className="flex-1 w-full h-full">
             {renderDetailChart()}
           </div>
        </div>

        {/* Right: Text Area */}
        <div className="w-full md:w-1/3 h-1/2 md:h-full p-8 md:p-12 flex flex-col justify-center border-l border-white/5 bg-white/[0.02]">
           <div className="mb-6">
              <span className="inline-block px-3 py-1 rounded-full bg-accent/10 text-accent text-xs font-bold uppercase tracking-wider mb-4">
                Visualization {currentIndex + 1} / 3
              </span>
              <h2 className="text-2xl font-bold text-white mb-4 leading-tight">
                {currentSlide.title}
              </h2>
              <div className="w-12 h-1 bg-accent mb-6 rounded-full"></div>
              <p className="text-slate-300 text-sm leading-relaxed">
                {currentSlide.description}
              </p>
           </div>
        </div>
      </div>
    </div>
  );
};


// --- HERO SECTION ---
const HeroSection = () => {
  const [activeVisualization, setActiveVisualization] = useState(0);
  const [chartData, setChartData] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // --- FETCH REAL DATA ON LOAD ---
  useEffect(() => {
    fetch('http://127.0.0.1:8000/hero-charts')
      .then(res => res.json())
      .then(data => setChartData(data))
      .catch(err => console.error("Failed to load hero charts:", err));
  }, []);

  // --- VIZ CONFIGURATION ---
  const visualizations = [
    {
      id: 1,
      title: "TGCA Test-KM by Cox Median Hazard Split",
      description: "Mechanistic hazard estimation with transparent modeling",
      accuracy: "0.791",
      dataPoints: 1059,
      color: "#ec4899", 
      type: "km",
      dataKey: "tcga_km"
    },
    {
      id: 2,
      title: "Metabric-KM by Cox Median Hazard Split",
      description: "Nonlinear ensemble risk pattern capture",
      accuracy: "0.632",
      dataPoints: 1903,
      color: "#3b82f6", 
      type: "km",
      dataKey: "metabric_km"
    },
    {
      id: 3,
      title: "Cox vs RSF Agreement-TCGA Test",
      description: "Combined Cox + RSF for robust survival prediction",
      accuracy: "0.850",
      dataPoints: 212,
      color: "#10b981", 
      type: "scatter",
      dataKey: "scatter"
    }
  ];

  // --- CAROUSEL LOGIC ---
  useEffect(() => {
    // Stop auto-rotation if modal is open
    if (isModalOpen) return;

    const interval = setInterval(() => {
      setActiveVisualization((prev) => (prev + 1) % visualizations?.length);
    }, 5000);
    return () => clearInterval(interval);
  }, [isModalOpen]); // Depend on isModalOpen

  const currentViz = visualizations?.[activeVisualization];
  const currentData = chartData ? chartData[currentViz.dataKey] : [];

  // --- CHART RENDERER (MINI) ---
  const renderChart = () => {
    if (!chartData) return <div className="h-full flex items-center justify-center text-muted-foreground text-sm">Loading Analytics...</div>;

    if (currentViz.type === "km") {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.1} vertical={false} />
            <XAxis dataKey="time" hide={false} axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b' }} label={{ value: 'Time', position: 'insideBottom', offset: -10, fontSize: 10, fill: '#64748b' }} />
            <YAxis hide={false} axisLine={false} tickLine={false} domain={[0, 1]} tick={{ fontSize: 10, fill: '#64748b' }} label={{ value: 'Survival Probability', angle: -90, position: 'insideLeft', offset: 15, fontSize: 10, fill: '#64748b' }} />
            <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px', fontSize: '12px' }} itemStyle={{ padding: 0 }} />
            <Legend verticalAlign="top" height={36} iconSize={8} wrapperStyle={{ fontSize: '12px' }}/>
            <Line type="stepAfter" data={currentData} dataKey="Low" name="Low Risk" stroke={currentViz.color} strokeWidth={3} dot={false} animationDuration={1500} />
            <Line type="stepAfter" data={currentData} dataKey="High" name="High Risk" stroke="#64748b" strokeWidth={2} strokeDasharray="4 4" dot={false} animationDuration={1500} />
          </LineChart>
        </ResponsiveContainer>
      );
    }

    return (
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
          <XAxis type="number" dataKey="x" name="Cox Hazard" tick={{ fontSize: 10, fill: '#64748b' }} tickLine={false} axisLine={false} label={{ value: 'CoxPH Hazard', position: 'insideBottom', offset: -10, fontSize: 10, fill: '#64748b' }} />
          <YAxis type="number" dataKey="y" name="RSF Risk" tick={{ fontSize: 10, fill: '#64748b' }} tickLine={false} axisLine={false} label={{ value: 'RSF Risk', angle: -90, position: 'insideLeft', offset: 15, fontSize: 10, fill: '#64748b' }} />
          <ZAxis range={[30, 30]} />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px', fontSize: '12px' }} />
          <Scatter name="Patients" data={currentData} fill={currentViz.color} opacity={0.6} animationDuration={1500} />
        </ScatterChart>
      </ResponsiveContainer>
    );
  };

  return (
    <>
      <section className="relative bg-background pt-32 pb-16 lg:pt-40 lg:pb-24 overflow-hidden">
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-20 left-10 w-64 h-64 bg-accent rounded-full blur-3xl animate-pulse-subtle"></div>
          <div className="absolute bottom-20 right-10 w-96 h-96 bg-primary rounded-full blur-3xl animate-pulse-subtle animation-delay-200"></div>
        </div>
        <div className="container mx-auto px-4 md:px-6 lg:px-8 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center">
            
            {/* Left Content */}
            <div className="space-y-6 md:space-y-8">
              <div className="inline-flex items-center space-x-2 px-4 py-2 bg-accent/10 rounded-full border border-accent/20">
                <Icon name="Award" size={16} color="var(--color-accent)" />
                <span className="text-xs md:text-sm font-medium text-accent">Research Project by Nandini</span>
              </div>

              <h1 className="text-3xl md:text-4xl lg:text-5xl xl:text-6xl font-bold text-foreground leading-tight">
                OncoRisk
                <span className="block text-accent mt-2">Clinicogenomic Survival Intelligence</span>
              </h1>

              <p className="text-base md:text-lg lg:text-xl text-muted-foreground leading-relaxed">
                Explore survival analysis, individual risk profiles, and biological drivers across breast cancer cohorts. Integrates clinical, pathological, and molecular data using a dual-model system designed for transparent hazard modeling and cohort-level profiling.
              </p>

              <div className="flex flex-col sm:flex-row gap-4">
                <Button variant="default" size="lg" iconName="FlaskConical" iconPosition="left" className="w-full sm:w-auto">Explore the Model</Button>
                <Button variant="outline" size="lg" iconName="BookOpen" iconPosition="left" className="w-full sm:w-auto">View Documentation</Button>
              </div>

              <div className="grid grid-cols-3 gap-4 md:gap-6 pt-6 border-t border-border">
                <div className="space-y-1">
                  <div className="text-2xl md:text-3xl lg:text-4xl font-bold text-accent">0.791</div>
                  <div className="text-xs md:text-sm text-muted-foreground">Cox Test C-index</div>
                </div>
                <div className="space-y-1">
                  <div className="text-2xl md:text-3xl lg:text-4xl font-bold text-primary">31</div>
                  <div className="text-xs md:text-sm text-muted-foreground">Gene Panel</div>
                </div>
                <div className="space-y-1">
                  <div className="text-2xl md:text-3xl lg:text-4xl font-bold text-success">2,962</div>
                  <div className="text-xs md:text-sm text-muted-foreground">Total Patients</div>
                </div>
              </div>
            </div>

            {/* Right Content - Interactive Card with Click Trigger */}
            <div className="relative group cursor-pointer" onClick={() => setIsModalOpen(true)}>
              {/* Hover Glow Effect */}
              <div className="absolute inset-0 bg-accent/20 blur-xl rounded-2xl opacity-0 group-hover:opacity-50 transition-opacity duration-500"></div>

              <div className="bg-card border border-border rounded-2xl p-6 md:p-8 shadow-2xl relative transition-transform duration-300 group-hover:scale-[1.01]">
                
                {/* Maximize Icon on Hover */}
                <div className="absolute top-4 right-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity">
                   <Maximize2 size={20} />
                </div>

                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg md:text-xl font-semibold text-foreground pr-8">{currentViz?.title}</h3>
                  <div className="flex items-center space-x-2 px-3 py-1 bg-success/10 rounded-full shrink-0">
                    <div className="w-2 h-2 bg-success rounded-full animate-pulse"></div>
                    <span className="text-xs font-medium text-success">Live</span>
                  </div>
                </div>

                <p className="text-sm text-muted-foreground mb-6 line-clamp-2">{currentViz?.description}</p>

                <div className="relative h-64 bg-secondary/20 rounded-xl overflow-hidden mb-6 border border-border/50 p-2">
                   {renderChart()}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-secondary/50 rounded-lg p-4">
                    <div className="text-xs text-muted-foreground mb-1">Accuracy Rate</div>
                    <div className="text-xl md:text-2xl font-bold text-foreground">{currentViz?.accuracy}</div>
                  </div>
                  <div className="bg-secondary/50 rounded-lg p-4">
                    <div className="text-xs text-muted-foreground mb-1">Data Points</div>
                    <div className="text-xl md:text-2xl font-bold text-foreground">{currentViz?.dataPoints?.toLocaleString()}</div>
                  </div>
                </div>

                <div className="flex items-center justify-center space-x-2 mt-6" onClick={(e) => e.stopPropagation()}>
                  {visualizations?.map((_, index) => (
                    <button
                      key={index}
                      onClick={() => setActiveVisualization(index)}
                      className={`w-2 h-2 rounded-full transition-all duration-300 ${
                        index === activeVisualization ? 'bg-accent w-8' : 'bg-muted'
                      }`}
                      aria-label={`View visualization ${index + 1}`}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* MODAL POPUP */}
      <CohortModal 
        isOpen={isModalOpen} 
        onClose={() => setIsModalOpen(false)} 
        initialIndex={activeVisualization}
        chartData={chartData}
      />
    </>
  );
};

export default HeroSection;