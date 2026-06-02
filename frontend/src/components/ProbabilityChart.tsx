"use client";

import { motion } from "framer-motion";

interface ProbabilityChartProps {
  probabilities: Record<string, number>;
  colors?: Record<string, string>;
}

export default function ProbabilityChart({ probabilities, colors }: ProbabilityChartProps) {
  // Default colors if none provided
  const defaultColors: Record<string, string> = {
    "No DR": "#10b981", // emerald
    "Mild": "#f59e0b", // amber
    "Moderate": "#f97316", // orange
    "Severe": "#ef4444", // red
    "Proliferative DR": "#991b1b", // dark red
  };

  const activeColors = colors || defaultColors;

  return (
    <div className="space-y-3 w-full">
      {Object.entries(probabilities).map(([className, prob], index) => {
        const percentage = Math.max(0.5, prob * 100); // Minimum 0.5% for visibility
        const color = activeColors[className] || "#6b7280";

        return (
          <div key={className} className="relative">
            <div className="flex justify-between items-end mb-1">
              <span className="text-xs font-medium text-gray-300">{className}</span>
              <span className="text-xs font-mono text-gray-400">
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-2.5 w-full bg-gray-800/50 rounded-full overflow-hidden border border-white/5">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${percentage}%` }}
                transition={{ duration: 0.8, delay: index * 0.1, ease: "easeOut" }}
                className="h-full rounded-full"
                style={{ backgroundColor: color }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
