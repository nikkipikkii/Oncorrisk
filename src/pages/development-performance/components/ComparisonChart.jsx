import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import Icon from '../../../components/AppIcon';

const ComparisonChart = ({ data, title, description }) => {
  const [activeMetric, setActiveMetric] = useState('accuracy');

  const metrics = [
    { key: 'accuracy', label: 'Accuracy', icon: 'Target', color: 'var(--color-accent)' },
    { key: 'sensitivity', label: 'Sensitivity', icon: 'Activity', color: 'var(--color-success)' },
    { key: 'specificity', label: 'Specificity', icon: 'Shield', color: 'var(--color-warning)' },
    { key: 'auc', label: 'AUC-ROC', icon: 'TrendingUp', color: 'var(--color-primary)' }
  ];

  const chartData = data?.map(item => ({
    name: item?.model,
    value: item?.[activeMetric] * 100
  }));

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload?.length) {
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-semibold text-foreground mb-1">{payload?.[0]?.payload?.name}</p>
          <p className="text-sm text-muted-foreground">
            {metrics?.find(m => m?.key === activeMetric)?.label}: 
            <span className="font-semibold text-accent ml-1">
              {payload?.[0]?.value?.toFixed(2)}%
            </span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-card rounded-lg border border-border p-4 md:p-6">
      <div className="mb-6">
        <h3 className="text-lg md:text-xl font-semibold text-foreground mb-2">{title}</h3>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      <div className="flex flex-wrap gap-2 mb-6">
        {metrics?.map((metric) => (
          <button
            key={metric?.key}
            onClick={() => setActiveMetric(metric?.key)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
              activeMetric === metric?.key
                ? 'bg-accent text-accent-foreground'
                : 'bg-muted text-muted-foreground hover:bg-muted/80'
            }`}
          >
            <Icon name={metric?.icon} size={16} />
            <span>{metric?.label}</span>
          </button>
        ))}
      </div>
      <div className="w-full h-80 md:h-96" aria-label={`${title} Bar Chart`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis 
              dataKey="name" 
              stroke="var(--color-muted-foreground)"
              angle={-45}
              textAnchor="end"
              height={80}
              tick={{ fill: 'var(--color-muted-foreground)', fontSize: 12 }}
            />
            <YAxis 
              stroke="var(--color-muted-foreground)"
              domain={[0, 100]}
              tick={{ fill: 'var(--color-muted-foreground)', fontSize: 12 }}
              label={{ value: 'Performance (%)', angle: -90, position: 'insideLeft', fill: 'var(--color-muted-foreground)' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend 
              wrapperStyle={{ paddingTop: '20px' }}
              iconType="circle"
            />
            <Bar 
              dataKey="value" 
              name={metrics?.find(m => m?.key === activeMetric)?.label}
              radius={[8, 8, 0, 0]}
            >
              {chartData?.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={entry?.name === 'OncoRisk' ? 'var(--color-accent)' : 'var(--color-muted)'} 
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
        {metrics?.map((metric) => {
          const oncoRiskValue = data?.find(d => d?.model === 'OncoRisk')?.[metric?.key] || 0;
          return (
            <div key={metric?.key} className="bg-muted/50 rounded-lg p-3">
              <div className="flex items-center space-x-2 mb-1">
                <Icon name={metric?.icon} size={14} color={metric?.color} />
                <span className="text-xs text-muted-foreground">{metric?.label}</span>
              </div>
              <p className="text-lg font-semibold text-foreground">
                {(oncoRiskValue * 100)?.toFixed(2)}%
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ComparisonChart;