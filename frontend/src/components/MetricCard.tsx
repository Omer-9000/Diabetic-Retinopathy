"use client";

import { ReactNode } from "react";

interface MetricCardProps {
  title: string;
  value: string | number;
  icon: ReactNode;
  trend?: string;
  trendUp?: boolean;
  subtitle?: string;
}

export default function MetricCard({ title, value, icon, trend, trendUp, subtitle }: MetricCardProps) {
  return (
    <div className="glass-card p-6 relative overflow-hidden group">
      {/* Background decoration */}
      <div className="absolute -right-6 -top-6 w-24 h-24 bg-sky-500/10 rounded-full blur-2xl group-hover:bg-sky-500/20 transition-colors" />
      
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-gray-400 mb-1">{title}</p>
          <h4 className="text-3xl font-bold text-white tracking-tight">{value}</h4>
          
          {(trend || subtitle) && (
            <div className="flex items-center mt-3">
              {trend && (
                <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                  trendUp === true ? 'bg-emerald-500/10 text-emerald-400' :
                  trendUp === false ? 'bg-red-500/10 text-red-400' :
                  'bg-gray-500/10 text-gray-400'
                }`}>
                  {trend}
                </span>
              )}
              {subtitle && (
                <span className={`text-xs text-gray-500 ${trend ? 'ml-2' : ''}`}>
                  {subtitle}
                </span>
              )}
            </div>
          )}
        </div>
        
        <div className="p-3 bg-gray-900/50 border border-gray-800 rounded-xl text-gray-400 group-hover:text-sky-400 group-hover:border-sky-500/30 transition-all shadow-inner">
          {icon}
        </div>
      </div>
    </div>
  );
}
