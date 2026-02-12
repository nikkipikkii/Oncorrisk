import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';
import Input from '../../../components/ui/Input';
import Select from '../../../components/ui/Select';

const ModelDemo = () => {
  const [demoStep, setDemoStep] = useState('input');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [inputMode, setInputMode] = useState('manual');

  const lymphNodeOptions = [
    { value: 'negative', label: 'Negative (NODE_POS = 0)' },
    { value: 'positive', label: 'Positive (NODE_POS > 0)' }
  ];

  const tcgaPatients = [
    { value: 'tcga_001', label: 'TCGA Patient #001 - Age 52, Node Negative' },
    { value: 'tcga_002', label: 'TCGA Patient #002 - Age 67, Node Positive' },
    { value: 'tcga_003', label: 'TCGA Patient #003 - Age 45, Node Negative' }
  ];

  const handleAnalyze = () => {
    setIsAnalyzing(true);
    setTimeout(() => {
      setIsAnalyzing(false);
      setDemoStep('results');
    }, 3000);
  };

  const handleReset = () => {
    setDemoStep('input');
  };

  const mockResults = {
    coxMedianSurvival: "68.2 months",
    rsfMedianSurvival: "72.5 months",
    consensusSurvival: "70.4 months",
    coxCIndex: "0.791",
    rsfCIndex: "0.736",
    keyFactors: [
      { name: "AGE", impact: "52 years", weight: "Clinical" },
      { name: "NODE_POS", impact: "Negative", weight: "Clinical" },
      { name: "31-Gene Panel", impact: "Z-scored expression", weight: "Molecular" }
    ],
    survivalCurveNote: "Median survival calculated at S(t) ≤ 0.5. RMST computed as area under survival curve."
  };

  return (
    <section className="py-16 md:py-20 lg:py-24 bg-background">
      <div className="container mx-auto px-4 md:px-6 lg:px-8">
        <div className="text-center mb-12 md:mb-16">
          <div className="inline-flex items-center space-x-2 px-4 py-2 bg-success/10 rounded-full border border-success/20 mb-4">
            <Icon name="FlaskConical" size={16} color="var(--color-success)" />
            <span className="text-xs md:text-sm font-medium text-success">Patient Risk Profile Inference</span>
          </div>
          <h2 className="text-2xl md:text-3xl lg:text-4xl font-bold text-foreground mb-4">
            Dual-Model Risk Profiling (Cox + RSF)
          </h2>
          <p className="text-base md:text-lg text-muted-foreground max-w-3xl mx-auto">
            Input clinical and molecular features to generate twin survival predictions using both Cox proportional hazards and Random Survival Forest models.
          </p>
        </div>

        <div className="max-w-5xl mx-auto">
          <div className="bg-card border border-border rounded-2xl overflow-hidden shadow-2xl">
            <div className="bg-secondary/50 border-b border-border p-4 md:p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
                    <Icon name="Activity" size={20} color="var(--color-accent)" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-foreground">OncoRisk Research Model</div>
                    <div className="text-xs text-muted-foreground">Cox + RSF Dual Inference System</div>
                  </div>
                </div>
                <div className="flex items-center space-x-2 px-3 py-1 bg-success/10 rounded-full">
                  <div className="w-2 h-2 bg-success rounded-full animate-pulse"></div>
                  <span className="text-xs font-medium text-success">Ready</span>
                </div>
              </div>
            </div>

            {demoStep === 'input' && (
              <div className="p-6 md:p-8 space-y-6">
                <div className="flex items-center space-x-4 mb-6">
                  <button
                    onClick={() => setInputMode('manual')}
                    className={`flex-1 py-3 px-4 rounded-lg border-2 transition-all ${
                      inputMode === 'manual' ?'border-accent bg-accent/10 text-accent font-semibold' :'border-border bg-secondary/30 text-muted-foreground hover:border-accent/50'
                    }`}
                  >
                    <div className="flex items-center justify-center space-x-2">
                      <Icon name="Edit3" size={18} />
                      <span className="text-sm">Manual Profile</span>
                    </div>
                  </button>
                  <button
                    onClick={() => setInputMode('tcga')}
                    className={`flex-1 py-3 px-4 rounded-lg border-2 transition-all ${
                      inputMode === 'tcga' ?'border-accent bg-accent/10 text-accent font-semibold' :'border-border bg-secondary/30 text-muted-foreground hover:border-accent/50'
                    }`}
                  >
                    <div className="flex items-center justify-center space-x-2">
                      <Icon name="Database" size={18} />
                      <span className="text-sm">Select TCGA Patient</span>
                    </div>
                  </button>
                </div>

                {inputMode === 'manual' ? (
                  <>
                    <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mb-6">
                      <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center space-x-2">
                        <Icon name="Stethoscope" size={16} color="var(--color-primary)" />
                        <span>Clinical Inputs</span>
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <Input
                          label="Age (years)"
                          type="number"
                          placeholder="Enter patient age"
                          description="Age at diagnosis"
                        />
                        <Select
                          label="Lymph Node Status"
                          options={lymphNodeOptions}
                          placeholder="Select status"
                          description="NODE_POS indicator"
                        />
                      </div>
                    </div>

                    <div className="bg-accent/5 border border-accent/20 rounded-lg p-4">
                      <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center space-x-2">
                        <Icon name="Dna" size={16} color="var(--color-accent)" />
                        <span>Molecular Profile (optional; default = 0)</span>
                      </h3>
                      <p className="text-xs text-muted-foreground mb-4">
                        Gene expression inputs (z-scored / approximate). The 31-gene panel includes curated markers for comprehensive risk assessment.
                      </p>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-h-48 overflow-y-auto">
                        {Array.from({ length: 31 }, (_, i) => (
                          <Input
                            key={i}
                            label={`Gene ${i + 1}`}
                            type="number"
                            placeholder="0.0"
                            step="0.1"
                          />
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="space-y-4">
                    <Select
                      label="Select Existing TCGA Test Patient"
                      options={tcgaPatients}
                      placeholder="Choose a patient from TCGA cohort"
                      description="Pre-loaded patient profiles from TCGA test set (n=1059)"
                    />
                    <div className="bg-secondary/50 rounded-lg p-4">
                      <div className="flex items-start space-x-3">
                        <Icon name="Info" size={20} color="var(--color-primary)" className="mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="text-sm text-muted-foreground">
                            These are anonymized patient profiles from the TCGA-BRCA test cohort, including clinical features (AGE, NODE_POS) and 31-gene molecular panel data.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <div className="bg-warning/10 border border-warning/20 rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <Icon name="AlertCircle" size={20} color="var(--color-warning)" className="mt-0.5 flex-shrink-0" />
                    <div>
                      <div className="text-sm font-semibold text-foreground mb-1">Research Platform Notice</div>
                      <p className="text-xs text-muted-foreground">
                        This platform is for research and stratification. It is not a regulated clinical decision system. Results are generated for demonstration and research purposes only.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex flex-col sm:flex-row gap-4 pt-4">
                  <Button
                    variant="default"
                    size="lg"
                    iconName="Play"
                    iconPosition="left"
                    onClick={handleAnalyze}
                    loading={isAnalyzing}
                    className="flex-1"
                  >
                    {isAnalyzing ? 'Running Cox + RSF Models...' : 'Run Dual-Model Analysis'}
                  </Button>
                  <Button
                    variant="outline"
                    size="lg"
                    iconName="RotateCcw"
                    iconPosition="left"
                    className="flex-1 sm:flex-initial"
                  >
                    Reset Form
                  </Button>
                </div>
              </div>
            )}

            {demoStep === 'results' && (
              <div className="p-6 md:p-8 space-y-6">
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center space-x-3">
                    <div className="w-12 h-12 rounded-full bg-success/10 flex items-center justify-center">
                      <Icon name="CheckCircle2" size={24} color="var(--color-success)" />
                    </div>
                    <div>
                      <div className="text-lg font-semibold text-foreground">Twin Inference Complete</div>
                      <div className="text-sm text-muted-foreground">Cox + RSF predictions generated</div>
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    iconName="Download"
                    iconPosition="left"
                  >
                    Export Report
                  </Button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
                  <div className="bg-accent/10 border border-accent/20 rounded-xl p-6 text-center">
                    <div className="text-sm text-accent font-medium mb-2">Cox Median Survival</div>
                    <div className="text-3xl md:text-4xl font-bold text-accent mb-2">{mockResults?.coxMedianSurvival}</div>
                    <div className="text-xs text-muted-foreground">C-index: {mockResults?.coxCIndex}</div>
                  </div>

                  <div className="bg-primary/10 border border-primary/20 rounded-xl p-6 text-center">
                    <div className="text-sm text-primary font-medium mb-2">RSF Median Survival</div>
                    <div className="text-3xl md:text-4xl font-bold text-primary mb-2">{mockResults?.rsfMedianSurvival}</div>
                    <div className="text-xs text-muted-foreground">C-index: {mockResults?.rsfCIndex}</div>
                  </div>

                  <div className="bg-success/10 border border-success/20 rounded-xl p-6 text-center">
                    <div className="text-sm text-success font-medium mb-2">Consensus Survival</div>
                    <div className="text-3xl md:text-4xl font-bold text-success mb-2">{mockResults?.consensusSurvival}</div>
                    <div className="text-xs text-muted-foreground">Average of Cox + RSF</div>
                  </div>
                </div>

                <div className="bg-secondary/50 rounded-xl p-6">
                  <h4 className="text-lg font-semibold text-foreground mb-4 flex items-center space-x-2">
                    <Icon name="TrendingUp" size={20} color="var(--color-accent)" />
                    <span>Input Features Used</span>
                  </h4>
                  <div className="space-y-3">
                    {mockResults?.keyFactors?.map((factor, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-card rounded-lg">
                        <div className="flex items-center space-x-3">
                          <div className={`w-2 h-2 rounded-full ${
                            factor?.weight === 'Clinical' ? 'bg-primary' : 'bg-accent'
                          }`}></div>
                          <div>
                            <div className="text-sm font-medium text-foreground">{factor?.name}</div>
                            <div className="text-xs text-muted-foreground">{factor?.impact}</div>
                          </div>
                        </div>
                        <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                          factor?.weight === 'Clinical' ? 'bg-primary/10 text-primary' : 'bg-accent/10 text-accent'
                        }`}>
                          {factor?.weight}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-secondary/50 rounded-xl p-6">
                  <h4 className="text-lg font-semibold text-foreground mb-4 flex items-center space-x-2">
                    <Icon name="Info" size={20} color="var(--color-primary)" />
                    <span>Survival Time Calculation</span>
                  </h4>
                  <div className="space-y-3">
                    <div className="flex items-start space-x-3 p-3 bg-card rounded-lg">
                      <Icon name="ArrowRight" size={16} color="var(--color-primary)" className="mt-0.5 flex-shrink-0" />
                      <p className="text-sm text-muted-foreground">{mockResults?.survivalCurveNote}</p>
                    </div>
                    <div className="flex items-start space-x-3 p-3 bg-card rounded-lg">
                      <Icon name="ArrowRight" size={16} color="var(--color-primary)" className="mt-0.5 flex-shrink-0" />
                      <p className="text-sm text-muted-foreground">
                        If neither curve crosses 0.5, median survival is reported as "not reached".
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex flex-col sm:flex-row gap-4 pt-4">
                  <Button
                    variant="default"
                    size="lg"
                    iconName="RotateCcw"
                    iconPosition="left"
                    onClick={handleReset}
                    className="flex-1"
                  >
                    Analyze Another Profile
                  </Button>
                  <Button
                    variant="outline"
                    size="lg"
                    iconName="BookOpen"
                    iconPosition="left"
                    className="flex-1"
                  >
                    View Methodology
                  </Button>
                </div>
              </div>
            )}
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-xl p-4 text-center">
              <Icon name="GitBranch" size={24} color="var(--color-accent)" className="mx-auto mb-2" />
              <div className="text-sm font-semibold text-foreground mb-1">Dual-Model System</div>
              <div className="text-xs text-muted-foreground">Cox + RSF twin inference</div>
            </div>
            <div className="bg-card border border-border rounded-xl p-4 text-center">
              <Icon name="Database" size={24} color="var(--color-primary)" className="mx-auto mb-2" />
              <div className="text-sm font-semibold text-foreground mb-1">2,962 Patients</div>
              <div className="text-xs text-muted-foreground">TCGA + METABRIC cohorts</div>
            </div>
            <div className="bg-card border border-border rounded-xl p-4 text-center">
              <Icon name="Activity" size={24} color="var(--color-success)" className="mx-auto mb-2" />
              <div className="text-sm font-semibold text-foreground mb-1">31-Gene Panel</div>
              <div className="text-xs text-muted-foreground">Curated molecular features</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ModelDemo;