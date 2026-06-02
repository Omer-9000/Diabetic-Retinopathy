"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Trophy, TrendingUp, Cpu, Activity, Clock, Layers, Medal, Zap } from "lucide-react";
import MetricCard from "@/components/MetricCard";
import ConfusionMatrix from "@/components/ConfusionMatrix";

export default function DashboardPage() {
  const [data, setData] = useState<any>(null);
  const [plots, setPlots] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortField, setSortField] = useState<string>("F1 Score");
  const [sortDesc, setSortDesc] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<"leaderboard" | "matrices" | "curves">("leaderboard");
  const [cmType, setCmType] = useState<"val" | "test">("test");

  useEffect(() => {
    Promise.all([
      fetch("http://localhost:5000/api/dashboard/leaderboard").then(res => res.json()),
      fetch("http://localhost:5000/api/dashboard/plots").then(res => res.json())
    ])
    .then(([leaderboardData, plotsData]) => {
      setData(leaderboardData);
      setPlots(plotsData.plots);
      setLoading(false);
    })
    .catch(err => {
      console.error("Failed to load dashboard data", err);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Activity className="w-8 h-8 text-sky-500 animate-spin" />
      </div>
    );
  }

  if (!data || !data.leaderboard || data.leaderboard.length === 0) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center text-center p-8">
        <Layers className="w-12 h-12 text-gray-700 mb-4" />
        <h2 className="text-2xl font-bold text-white mb-2">No Training Data Found</h2>
        <p className="text-gray-400">Run the training pipeline first to generate metrics and plots.</p>
      </div>
    );
  }

  const sortedLeaderboard = [...data.leaderboard].sort((a, b) => {
    const valA = a[sortField] || 0;
    const valB = b[sortField] || 0;
    return sortDesc ? valB - valA : valA - valB;
  });

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDesc(!sortDesc);
    } else {
      setSortField(field);
      setSortDesc(true);
    }
  };

  const getMedalColor = (index: number) => {
    if (index === 0) return "text-yellow-400";
    if (index === 1) return "text-gray-300";
    if (index === 2) return "text-amber-600";
    return "text-transparent";
  };

  const bestModel = sortedLeaderboard[0];

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-extrabold tracking-tight text-white mb-2">
          Model Performance Dashboard
        </h1>
        <p className="text-gray-400">
          Comparative analysis of 6 deep learning architectures trained on the APTOS 2019 dataset.
        </p>
      </div>

      {/* Top Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard 
          title="Top F1 Score" 
          value={(bestModel['F1 Score'] * 100).toFixed(1) + '%'} 
          icon={<Trophy />} 
          subtitle={bestModel['Model Name']}
        />
        <MetricCard 
          title="Best Accuracy" 
          value={(Math.max(...data.leaderboard.map((m: any) => m.Accuracy)) * 100).toFixed(1) + '%'} 
          icon={<Activity />} 
        />
        <MetricCard 
          title="Highest ROC-AUC" 
          value={(Math.max(...data.leaderboard.map((m: any) => m['ROC-AUC'])) * 100).toFixed(1) + '%'} 
          icon={<TrendingUp />} 
        />
        <MetricCard 
          title="Total Models" 
          value={data.leaderboard.length} 
          icon={<Layers />} 
        />
      </div>

      {/* Tabs */}
      <div className="flex space-x-1 bg-gray-900/50 p-1 rounded-xl w-fit border border-gray-800">
        {[
          { id: "leaderboard", label: "Leaderboard" },
          { id: "matrices", label: "Confusion Matrices" },
          { id: "curves", label: "Learning Curves" }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTab === tab.id
                ? "bg-sky-500/20 text-sky-400 shadow-sm"
                : "text-gray-400 hover:text-white hover:bg-gray-800"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        {activeTab === "leaderboard" && (
          <motion.div
            key="leaderboard"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="glass-card overflow-hidden"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-gray-400 uppercase bg-gray-900/80 border-b border-gray-800">
                  <tr>
                    <th className="px-6 py-4 w-12">Rank</th>
                    <th className="px-6 py-4 cursor-pointer hover:text-white" onClick={() => handleSort('Model Name')}>
                      Model
                    </th>
                    <th className="px-6 py-4 cursor-pointer hover:text-white" onClick={() => handleSort('F1 Score')}>
                      F1 Score
                    </th>
                    <th className="px-6 py-4 cursor-pointer hover:text-white" onClick={() => handleSort('Accuracy')}>
                      Accuracy
                    </th>
                    <th className="px-6 py-4 cursor-pointer hover:text-white" onClick={() => handleSort('ROC-AUC')}>
                      ROC-AUC
                    </th>
                    <th className="px-6 py-4 cursor-pointer hover:text-white" onClick={() => handleSort('Number of Parameters')}>
                      Params
                    </th>
                    <th className="px-6 py-4 cursor-pointer hover:text-white text-right" onClick={() => handleSort('Training Time (s)')}>
                      Train Time
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sortedLeaderboard.map((row, idx) => (
                    <tr 
                      key={row['Model Name']} 
                      className={`border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors ${
                        row['Model Name'] === data.best_model ? "bg-sky-950/20" : ""
                      }`}
                    >
                      <td className="px-6 py-4">
                        {idx < 3 ? (
                          <Medal className={`w-5 h-5 ${getMedalColor(idx)}`} />
                        ) : (
                          <span className="text-gray-500 font-mono ml-1">{idx + 1}</span>
                        )}
                      </td>
                      <td className="px-6 py-4 font-medium text-white">
                        {row['Model Name']}
                        {row['Model Name'] === data.best_model && (
                          <span className="ml-2 px-2 py-0.5 text-[10px] uppercase bg-sky-500/20 text-sky-400 rounded-full border border-sky-500/30">
                            Best
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4 font-mono">
                        <div className="flex items-center">
                          <span className={idx === 0 && sortField === 'F1 Score' ? "text-sky-400 font-bold" : "text-gray-300"}>
                            {(row['F1 Score'] * 100).toFixed(2)}%
                          </span>
                          <div className="ml-3 w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                            <div className="h-full bg-sky-500 rounded-full" style={{ width: `${row['F1 Score'] * 100}%` }} />
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 font-mono text-gray-300">
                        {(row['Accuracy'] * 100).toFixed(2)}%
                      </td>
                      <td className="px-6 py-4 font-mono text-gray-300">
                        {(row['ROC-AUC'] * 100).toFixed(2)}%
                      </td>
                      <td className="px-6 py-4 text-gray-400">
                        <div className="flex items-center space-x-1">
                          <Cpu className="w-3.5 h-3.5" />
                          <span>{(row['Number of Parameters'] / 1000000).toFixed(1)}M</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-gray-400 text-right">
                        <div className="flex items-center justify-end space-x-1">
                          <Clock className="w-3.5 h-3.5" />
                          <span>{Math.round(row['Training Time (s)'] / 60)}m {Math.round(row['Training Time (s)'] % 60)}s</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}

        {activeTab === "matrices" && (
          <motion.div
            key="matrices"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-6"
          >
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-white">Model Confusion Matrices</h3>
              <div className="flex space-x-2 bg-gray-900/50 p-1 rounded-lg border border-gray-800">
                <button
                  onClick={() => setCmType("val")}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                    cmType === "val" ? "bg-gray-700 text-white" : "text-gray-400 hover:text-white"
                  }`}
                >
                  Validation Set
                </button>
                <button
                  onClick={() => setCmType("test")}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                    cmType === "test" ? "bg-gray-700 text-white" : "text-gray-400 hover:text-white"
                  }`}
                >
                  Test Set
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {plots
                .filter(p => p.type === (cmType === "test" ? "test_confusion_matrix" : "confusion_matrix"))
                .map(plot => (
                  <ConfusionMatrix key={plot.filename} modelName={plot.model} type={cmType} url={plot.url} />
                ))}
            </div>
          </motion.div>
        )}

        {activeTab === "curves" && (
          <motion.div
            key="curves"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <h3 className="text-lg font-medium text-white mb-6">Training & Validation Learning Curves</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {plots
                .filter(p => p.type === "learning_curves")
                .map(plot => (
                  <div key={plot.filename} className="glass-card p-4">
                    <p className="text-sm font-medium text-gray-300 mb-3 text-center uppercase tracking-wider">{plot.model}</p>
                    <div className="relative aspect-[16/7] bg-white rounded-lg overflow-hidden border border-gray-800">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={`http://localhost:5000${plot.url}`} alt={`Learning Curves ${plot.model}`} className="w-full h-full object-fill" />
                    </div>
                  </div>
                ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
