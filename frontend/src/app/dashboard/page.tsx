"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Trophy, TrendingUp, Cpu, Clock, Activity, Target, Layers, Star } from "lucide-react";

export default function DashboardPage() {
  const [leaderboardData, setLeaderboardData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"base" | "finetune" | "advanced">("advanced");

  useEffect(() => {
    // In a real app this would fetch from an API
    // For now we'll simulate the data based on our approach_1 results
    const fetchLeaderboard = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/dashboard/leaderboard");
        if (response.ok) {
          const data = await response.json();
          setLeaderboardData(data);
        } else {
          throw new Error("API error");
        }
      } catch (error) {
        console.warn("Failed to fetch from API, falling back to static data", error);
        // Fallback to static data if API is not fully updated yet
        setLeaderboardData({
          best_model: "R2_mixup",
          base: [
            { name: "EfficientNet-V2-S", f1: 0.820, acc: 0.816, bal_acc: 0.789, kappa: 0.746, auc: null, params: 20.18, rank: 1 },
            { name: "EfficientNet-B3", f1: 0.771, acc: 0.754, bal_acc: 0.786, kappa: 0.672, auc: null, params: 10.7, rank: 2 },
            { name: "ConvNeXt-Tiny", f1: 0.771, acc: 0.763, bal_acc: 0.780, kappa: 0.681, auc: null, params: 27.82, rank: 3 },
            { name: "DenseNet-121", f1: 0.745, acc: 0.728, bal_acc: 0.788, kappa: 0.643, auc: null, params: 6.96, rank: 4 },
            { name: "Swin-T", f1: 0.548, acc: 0.553, bal_acc: 0.545, kappa: 0.403, auc: null, params: 27.52, rank: 5 },
            { name: "Custom CNN", f1: 0.510, acc: 0.518, bal_acc: 0.524, kappa: 0.358, auc: null, params: 5.14, rank: 6 }
          ],
          finetune: [
            { name: "V4: Full Fine-tune", f1: 0.790, acc: 0.781, bal_acc: 0.788, kappa: 0.702, auc: 0.948, rank: 1 },
            { name: "V5: Two-Stage", f1: 0.749, acc: 0.737, bal_acc: 0.719, kappa: 0.644, auc: 0.932, rank: 2 },
            { name: "V3: Label Smooth", f1: 0.664, acc: 0.658, bal_acc: 0.647, kappa: 0.534, auc: 0.856, rank: 3 },
            { name: "V0: Baseline", f1: 0.594, acc: 0.570, bal_acc: 0.585, kappa: 0.440, auc: 0.849, rank: 4 },
            { name: "V1: Deeper Head", f1: 0.543, acc: 0.535, bal_acc: 0.578, kappa: 0.405, auc: 0.862, rank: 5 },
            { name: "V2: Attn Pool", f1: 0.376, acc: 0.386, bal_acc: 0.395, kappa: 0.206, auc: 0.724, rank: 6 }
          ],
          advanced: [
            { name: "R2_mixup", f1: 0.917, acc: 0.912, bal_acc: 0.904, kappa: 0.879, auc: 0.973, rank: 1 },
            { name: "R1+R2_combined", f1: 0.862, acc: 0.860, bal_acc: 0.848, kappa: 0.806, auc: 0.970, rank: 2 },
            { name: "T1_threshold_opt", f1: 0.820, acc: 0.816, bal_acc: 0.791, kappa: 0.746, auc: 0.948, rank: 3 },
            { name: "R1_differential_lr", f1: 0.816, acc: 0.807, bal_acc: 0.807, kappa: 0.736, auc: 0.949, rank: 4 },
            { name: "T1+T2_combined", f1: 0.809, acc: 0.807, bal_acc: 0.770, kappa: 0.733, auc: 0.939, rank: 5 },
            { name: "R3_warmup", f1: 0.802, acc: 0.789, bal_acc: 0.803, kappa: 0.715, auc: 0.949, rank: 6 },
            { name: "V4_baseline", f1: 0.790, acc: 0.781, bal_acc: 0.788, kappa: 0.702, auc: 0.948, rank: 7 },
            { name: "T2_tta", f1: 0.779, acc: 0.772, bal_acc: 0.750, kappa: 0.686, auc: 0.939, rank: 8 }
          ]
        });
      } finally {
        setLoading(false);
      }
    };

    fetchLeaderboard();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Activity className="w-8 h-8 text-sky-500 animate-spin" />
      </div>
    );
  }

  // Use API structure if present, otherwise use our mock
  const isApiFormat = leaderboardData?.leaderboard !== undefined;
  
  // Format data helper
  const formatPercent = (val: number | null | undefined) => {
    if (val === null || val === undefined) return "N/A";
    return (val * 100).toFixed(1) + "%";
  };
  
  const formatDec = (val: number | null | undefined) => {
    if (val === null || val === undefined) return "N/A";
    return val.toFixed(4);
  };

  // Content rendering based on active tab
  const getTableData = () => {
    if (isApiFormat) {
      // In a real scenario we'd parse the single CSV from the API here
      // But since we know we need 3 phases and the backend might just have base models,
      // we'll rely on our static structure for the presentation layer if needed
      return leaderboardData.leaderboard || [];
    }
    return leaderboardData[activeTab];
  };

  const tableData = getTableData();

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-8">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white flex items-center gap-3">
            <Trophy className="w-8 h-8 text-amber-400" />
            Performance Leaderboard
          </h1>
          <p className="mt-2 text-gray-400 max-w-2xl text-sm">
            Evaluating model performance across 3 architectural phases on the DIP-enhanced dataset. Metrics are calculated using a strict independent test set.
          </p>
        </div>
        
        {/* Top Model Badge */}
        <div className="bg-gradient-to-r from-amber-500/10 to-orange-500/10 border border-amber-500/20 p-4 rounded-xl flex items-center gap-4 shrink-0">
          <div className="w-10 h-10 bg-amber-500/20 rounded-full flex items-center justify-center">
            <Star className="w-5 h-5 text-amber-400 fill-amber-400" />
          </div>
          <div>
            <p className="text-xs text-amber-500 font-bold uppercase tracking-wider">Overall Best</p>
            <p className="text-lg font-bold text-white">R2_mixup</p>
            <p className="text-xs text-amber-200">F1: 91.7% • AUC: 97.3%</p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      {!isApiFormat && (
        <div className="flex bg-gray-900/50 p-1 rounded-xl border border-gray-800 w-fit">
          <button
            onClick={() => setActiveTab("base")}
            className={`px-6 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === "base" ? "bg-sky-500/20 text-sky-400" : "text-gray-400 hover:text-white"
            }`}
          >
            Phase 1: Base Models
          </button>
          <button
            onClick={() => setActiveTab("finetune")}
            className={`px-6 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === "finetune" ? "bg-teal-500/20 text-teal-400" : "text-gray-400 hover:text-white"
            }`}
          >
            Phase 2: Fine-tune Variants
          </button>
          <button
            onClick={() => setActiveTab("advanced")}
            className={`px-6 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === "advanced" ? "bg-amber-500/20 text-amber-400" : "text-gray-400 hover:text-white"
            }`}
          >
            Phase 3: Advanced Optimisation
          </button>
        </div>
      )}

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card p-4 flex items-center gap-4 border-l-4 border-l-sky-500">
          <div className="p-3 bg-sky-500/10 rounded-lg text-sky-400"><Target className="w-5 h-5" /></div>
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Primary Metric</p>
            <p className="font-bold text-white text-sm">Macro F1 Score</p>
          </div>
        </div>
        <div className="glass-card p-4 flex items-center gap-4 border-l-4 border-l-emerald-500">
          <div className="p-3 bg-emerald-500/10 rounded-lg text-emerald-400"><Activity className="w-5 h-5" /></div>
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Calibration</p>
            <p className="font-bold text-white text-sm">ROC-AUC</p>
          </div>
        </div>
        <div className="glass-card p-4 flex items-center gap-4 border-l-4 border-l-purple-500">
          <div className="p-3 bg-purple-500/10 rounded-lg text-purple-400"><Layers className="w-5 h-5" /></div>
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Imbalance check</p>
            <p className="font-bold text-white text-sm">Balanced Accuracy</p>
          </div>
        </div>
        <div className="glass-card p-4 flex items-center gap-4 border-l-4 border-l-amber-500">
          <div className="p-3 bg-amber-500/10 rounded-lg text-amber-400"><TrendingUp className="w-5 h-5" /></div>
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Agreement</p>
            <p className="font-bold text-white text-sm">Cohen's Kappa</p>
          </div>
        </div>
      </div>

      {/* Leaderboard Table */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card overflow-hidden"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-gray-400 uppercase bg-gray-900/80 border-b border-gray-800">
              <tr>
                <th className="px-6 py-4 w-16">Rank</th>
                <th className="px-6 py-4">Model / Technique</th>
                <th className="px-6 py-4 text-sky-400">Macro F1</th>
                <th className="px-6 py-4">Balanced Acc</th>
                <th className="px-6 py-4">Accuracy</th>
                <th className="px-6 py-4">ROC-AUC</th>
                <th className="px-6 py-4">Kappa</th>
                {activeTab === "base" && <th className="px-6 py-4 text-right">Params (M)</th>}
              </tr>
            </thead>
            <tbody>
              {tableData.map((row: any, idx: number) => {
                // Parse API or mock data
                const rank = row.rank || row.Rank || idx + 1;
                const name = row.name || row["Model Name"] || row.Variant || row.Technique;
                const f1 = row.f1 !== undefined ? row.f1 : row["F1 Score"];
                const balAcc = row.bal_acc !== undefined ? row.bal_acc : row["Balanced Accuracy"];
                const acc = row.acc !== undefined ? row.acc : row["Accuracy"];
                const auc = row.auc !== undefined ? row.auc : row["ROC-AUC"];
                const kappa = row.kappa !== undefined ? row.kappa : row["Cohen Kappa"];
                const params = row.params !== undefined ? row.params : row["Params (M)"];

                const isTop = rank === 1;

                return (
                  <tr
                    key={name}
                    className={`border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors ${
                      isTop ? "bg-amber-950/10" : ""
                    }`}
                  >
                    <td className="px-6 py-4">
                      {isTop ? (
                        <div className="flex items-center justify-center w-6 h-6 rounded-full bg-amber-500/20 text-amber-400">
                          <Trophy className="w-3 h-3" />
                        </div>
                      ) : (
                        <span className="text-gray-500 font-mono pl-2">{rank}</span>
                      )}
                    </td>
                    <td className="px-6 py-4 font-medium text-white">
                      {name}
                    </td>
                    <td className={`px-6 py-4 font-mono font-bold ${isTop ? "text-amber-400" : "text-sky-400"}`}>
                      {formatPercent(f1)}
                    </td>
                    <td className="px-6 py-4 font-mono text-gray-300">
                      {formatPercent(balAcc)}
                    </td>
                    <td className="px-6 py-4 font-mono text-gray-400">
                      {formatPercent(acc)}
                    </td>
                    <td className="px-6 py-4 font-mono text-gray-300">
                      {formatPercent(auc)}
                    </td>
                    <td className="px-6 py-4 font-mono text-gray-400">
                      {formatDec(kappa)}
                    </td>
                    {activeTab === "base" && (
                      <td className="px-6 py-4 font-mono text-gray-500 text-right">
                        {params ? params.toFixed(2) : "N/A"}
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </motion.div>
      
      {/* Metric explanation */}
      <div className="bg-gray-900/50 p-5 rounded-xl border border-gray-800 text-sm text-gray-400 leading-relaxed">
        <strong>Why Macro F1?</strong> The dataset is highly imbalanced. Raw accuracy can be misleading (predicting the majority class artificially inflates accuracy). Macro F1 treats all classes equally, making it the most robust indicator of true clinical performance. We also report <strong>Balanced Accuracy</strong> (average of sensitivities across classes) and <strong>Cohen's Kappa</strong> (agreement beyond chance).
      </div>
    </div>
  );
}
