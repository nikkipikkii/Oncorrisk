import React from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';
import {
  ArrowRightLeft,
  ArrowUpRight,
  Plus
} from "lucide-react";

const HeroSection = () => {
  return (
    <section className="relative bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 pt-32 pb-16 lg:pt-40 lg:pb-24 overflow-hidden">
      
      {/* Background glow */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-20 left-10 w-64 h-64 bg-accent rounded-full blur-3xl"></div>
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-primary rounded-full blur-3xl"></div>
      </div>

      <div className="container mx-auto px-4 md:px-6 lg:px-8 relative z-10">
        <div className="max-w-4xl mx-auto text-center space-y-6 md:space-y-8">

          {/* Badge */}
          {/* <div className="inline-flex items-center space-x-2 px-4 py-2 bg-accent/10 border border-accent/20 rounded-full">
            <Icon name="Hospital" size={18} color="var(--color-accent)" />
            <span className="text-sm md:text-base text-accent font-medium">
              Healthcare System Integration
            </span>
          </div> */}

          {/* Title */}
          <h1 className="text-3xl md:text-4xl lg:text-6xl font-bold text-foreground leading-tight">
            Gene-Level Biological Intelligence with{' '}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent to-primary">
              Gene Narrative & Metrics
            </span>
          </h1>

          {/* Description */}
          <p className="text-base md:text-lg lg:text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            This view describes how the selected gene contributes to the Cox proportional hazards model. Each
            coefficient represents the direction and strength of association between gene expression and modeled
            hazard. Positive coefficients indicate association with increased hazard, while negative coefficients
            suggest association with reduced hazard. The magnitude reflects relative influence within the model,
            not standalone biological causality.
          </p>

          {/* ================= INSERTED GENE NARRATIVE BLOCK ================= */}
          <div className="max-w-3xl mx-auto mt-6 text-left">
            <div className="bg-slate-900/60 backdrop-blur border border-slate-800 rounded-xl px-6 py-6">
              
              <h3 className="text-lg md:text-xl font-semibold text-indigo-400 mb-3">
                Gene Narrative & Model Metrics
              </h3>

              <p className="text-base md:text-lg text-slate-300 mb-5 leading-relaxed">
                This view explains how the selected gene contributes to the Cox
                proportional hazards model.
              </p>

              <ol className="space-y-3 text-sm md:text-base text-slate-300">
                <li className="flex gap-3">
                  <ArrowRightLeft size={16} className="text-indigo-400 mt-1" />
                  Each coefficient represents direction and strength of association.
                </li>

                <li className="flex gap-3">
                  <Plus size={16} className="text-indigo-400 mt-1" />
                  Positive coefficients indicate increased hazard.
                </li>

                <li className="flex gap-3">
                  <ArrowUpRight size={16} className="text-indigo-400 mt-1" />
                  Magnitude reflects relative model influence.
                </li>
              </ol>
            </div>
          </div>
          {/* ================================================================ */}

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <Button variant="default" size="lg" iconName="Rocket" iconPosition="left">
              Go to Model Contribution
            </Button>
            {/* <Button variant="outline" size="lg" iconName="Calculator" iconPosition="left">
              Calculate ROI
            </Button> */}
          </div>

          {/* Stats
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8 pt-8 md:pt-12">
            
            <div className="bg-card/50 backdrop-blur-sm border border-border rounded-lg p-6">
              <div className="w-12 h-12 bg-success/10 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <Icon name="CheckCircle2" size={24} color="var(--color-success)" />
              </div>
              <h3 className="text-xl md:text-2xl font-bold text-foreground mb-2">98%</h3>
              <p className="text-sm text-muted-foreground">
                Implementation Success Rate
              </p>
            </div>

            <div className="bg-card/50 backdrop-blur-sm border border-border rounded-lg p-6">
              <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <Icon name="Clock" size={24} color="var(--color-accent)" />
              </div>
              <h3 className="text-xl md:text-2xl font-bold text-foreground mb-2">
                4–6 Weeks
              </h3>
              <p className="text-sm text-muted-foreground">
                Average Deployment Time
              </p>
            </div>

            <div className="bg-card/50 backdrop-blur-sm border border-border rounded-lg p-6">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <Icon name="Users" size={24} color="var(--color-primary)" />
              </div>
              <h3 className="text-xl md:text-2xl font-bold text-foreground mb-2">
                200+
              </h3>
              <p className="text-sm text-muted-foreground">
                Healthcare Systems Deployed
              </p>
            </div>

          </div> */}
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
