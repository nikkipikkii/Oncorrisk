import React from 'react';
import Icon from '../../../components/AppIcon';

const PerformanceMetricsCard = ({ metric }) => {
  const getMetricColor = (value) => {
    if (value >= 0.9) return 'text-success';
    if (value >= 0.8) return 'text-warning';
    return 'text-error';
  };

  const getIconColor = (value) => {
    if (value >= 0.9) return 'var(--color-success)';
    if (value >= 0.8) return 'var(--color-warning)';
    return 'var(--color-error)';
  };

  return (
    <div className="bg-card rounded-lg p-4 md:p-6 border border-border hover:border-accent transition-all duration-300">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 md:w-12 md:h-12 rounded-lg bg-primary/10 flex items-center justify-center">
            <Icon name={metric?.icon} size={20} color="var(--color-accent)" />
          </div>
          <div>
            <h3 className="text-base md:text-lg font-semibold text-foreground">{metric?.name}</h3>
            <p className="text-xs md:text-sm text-muted-foreground">{metric?.description}</p>
          </div>
        </div>
      </div>
      <div className="space-y-3">
        <div className="flex items-end justify-between">
          <span className={`text-3xl md:text-4xl font-bold ${getMetricColor(metric?.value)}`}>
            {(metric?.value * 100)?.toFixed(1)}%
          </span>
          <div className="flex items-center space-x-1">
            <Icon 
              name={metric?.trend === 'up' ? 'TrendingUp' : 'TrendingDown'} 
              size={16} 
              color={getIconColor(metric?.value)} 
            />
            <span className={`text-sm font-medium ${getMetricColor(metric?.value)}`}>
              {metric?.change}
            </span>
          </div>
        </div>
        
        <div className="w-full bg-muted rounded-full h-2">
          <div 
            className={`h-2 rounded-full transition-all duration-500 ${
              metric?.value >= 0.9 ? 'bg-success' : metric?.value >= 0.8 ? 'bg-warning' : 'bg-error'
            }`}
            style={{ width: `${metric?.value * 100}%` }}
          />
        </div>
        
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Baseline: {(metric?.baseline * 100)?.toFixed(1)}%</span>
          <span>Target: {(metric?.target * 100)?.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
};

export default PerformanceMetricsCard;