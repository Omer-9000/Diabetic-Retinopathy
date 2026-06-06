"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, LineChart, Brain, Target, Star, Network, FlaskConical, BarChart3, Gauge } from "lucide-react";
import AuthGuard from "@/components/AuthGuard";
import { API_BASE, authFetch } from "@/lib/auth";

type Phase = "base" | "finetune" | "advanced";

interface PlotEntry {
  filename: string;
  type: string;
  model: string;
  phase: string;
  url: string;
}

function TrainingContent() {
  const [plots, setPlots] = useState<PlotEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [activePhase, setActivePhase] = useState<Phase>("advanced");
  const [activeModel, setActiveModel] = useState<string>("R2_mixup");

  // Define the models for each phase with verified metrics from actual results
  const phasesData = {
    base: [
      { name: "efficientnet_v2_s", display_name: "EfficientNet-V2-S", is_best: true, acc: 0.8158, f1: 0.8200, auc: null },
      { name: "efficientnet_b3", display_name: "EfficientNet-B3", is_best: false, acc: 0.7544, f1: 0.7709, auc: null },
      { name: "convnext_tiny", display_name: "ConvNeXt-Tiny", is_best: false, acc: 0.7632, f1: 0.7706, auc: null },
      { name: "densenet121", display_name: "DenseNet-121", is_best: false, acc: 0.7281, f1: 0.7452, auc: null },
      { name: "swin_t", display_name: "Swin-T", is_best: false, acc: 0.5526, f1: 0.5475, auc: null },
      { name: "custom_cnn", display_name: "Custom CNN", is_best: false, acc: 0.5175, f1: 0.5102, auc: null }
    ],
    finetune: [
      { name: "V4_full_finetune", display_name: "V4: Full Fine-tune", is_best: true, acc: 0.7807, f1: 0.7903, auc: 0.9483 },
      { name: "V5_two_stage", display_name: "V5: Two-Stage", is_best: false, acc: 0.7368, f1: 0.7491, auc: 0.9319 },
      { name: "V3_label_smooth", display_name: "V3: Label Smooth", is_best: false, acc: 0.6579, f1: 0.6639, auc: 0.8557 },
      { name: "V0_baseline", display_name: "V0: Baseline", is_best: false, acc: 0.5702, f1: 0.5942, auc: 0.8486 },
      { name: "V1_deeper_head", display_name: "V1: Deeper Head", is_best: false, acc: 0.5351, f1: 0.5431, auc: 0.8615 },
      { name: "V2_attention_pool", display_name: "V2: Attn Pool", is_best: false, acc: 0.3860, f1: 0.3763, auc: 0.7243 }
    ],
    advanced: [
      { name: "R2_mixup", display_name: "R2: MixUp", is_best: true, acc: 0.9123, f1: 0.9174, auc: 0.9731 },
      { name: "R1+R2_combined", display_name: "R1+R2: Combined", is_best: false, acc: 0.8596, f1: 0.8620, auc: 0.9699 },
      { name: "R1_differential_lr", display_name: "R1: Differential LR", is_best: false, acc: 0.8070, f1: 0.8162, auc: 0.9494 },
      { name: "R3_warmup", display_name: "R3: Warmup", is_best: false, acc: 0.7895, f1: 0.8016, auc: 0.9491 }
    ]
  };

  useEffect(() => {
    // Fetch plots from the API
    authFetch(`${API_BASE}/api/dashboard/plots`)
      .then((res) => res.json())
      .then((plotsData) => {
        setPlots(plotsData.plots || []);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load training data", err);
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

  // Handle phase change
  const handlePhaseChange = (phase: Phase) => {
    setActivePhase(phase);
    setActiveModel(phasesData[phase][0].name);
  };

  const currentModels = phasesData[activePhase];
  const activeModelData = currentModels.find((m) => m.name === activeModel);

  // Filter plots for the active model — match by model name in the plot entry
  const modelPlots = plots.filter((p) => p.model === activeModel);
  
  // Find specific plot types
  const learningCurve = modelPlots.find((p) => p.type === "learning_curves");
  const confusionMatrix = modelPlots.find((p) => p.type === "test_confusion_matrix");
  const perClassBar = modelPlots.find((p) => p.type === "per_class");
  const calibrationPlot = modelPlots.find((p) => p.type === "calibration");

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-10">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <div className="w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg shadow-indigo-500/20">
          <LineChart className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight text-white mb-4">
          Training Analytics
        </h1>
        <p className="text-lg text-gray-400">
          Detailed learning curves, confusion matrices, per-class metrics, and calibration plots across all 3 phases of model development.
        </p>
      </div>

      {/* Phase Selector */}
      <div className="flex bg-gray-900/50 p-1 rounded-xl border border-gray-800 w-fit mx-auto">
        <button
          onClick={() => handlePhaseChange("base")}
          className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium transition-colors ${
            activePhase === "base" ? "bg-sky-500/20 text-sky-400" : "text-gray-400 hover:text-white"
          }`}
        >
          <Network className="w-4 h-4" /> Phase 1: Base Models
        </button>
        <button
          onClick={() => handlePhaseChange("finetune")}
          className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium transition-colors ${
            activePhase === "finetune" ? "bg-teal-500/20 text-teal-400" : "text-gray-400 hover:text-white"
          }`}
        >
          <FlaskConical className="w-4 h-4" /> Phase 2: Fine-tune
        </button>
        <button
          onClick={() => handlePhaseChange("advanced")}
          className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium transition-colors ${
            activePhase === "advanced" ? "bg-amber-500/20 text-amber-400" : "text-gray-400 hover:text-white"
          }`}
        >
          <Star className="w-4 h-4" /> Phase 3: Advanced
        </button>
      </div>

      {/* Model Selector Tabs */}
      <div className="flex flex-wrap justify-center gap-2 mb-8">
        {currentModels.map((model) => (
          <button
            key={model.name}
            onClick={() => setActiveModel(model.name)}
            className={`px-5 py-2.5 rounded-xl text-sm font-medium transition-all ${
              activeModel === model.name
                ? "bg-gradient-to-r from-gray-800 to-gray-700 text-white border border-gray-500 shadow-lg shadow-black/20"
                : "bg-gray-900/50 text-gray-400 border border-gray-800 hover:bg-gray-800 hover:text-white"
            }`}
          >
            {model.display_name}
            {model.is_best && (
              <span className={`ml-2 px-1.5 py-0.5 text-[10px] uppercase rounded border ${
                activePhase === "base" ? "bg-sky-500/20 text-sky-400 border-sky-500/30" :
                activePhase === "finetune" ? "bg-teal-500/20 text-teal-400 border-teal-500/30" :
                "bg-amber-500/20 text-amber-400 border-amber-500/30"
              }`}>
                {activePhase === "advanced" ? "Final" : "Selected"}
              </span>
            )}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {activeModel && activeModelData && (
          <motion.div
            key={activeModel}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="space-y-8"
          >
            {/* Quick Stats for Active Model */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="glass-card p-4 flex flex-col items-center justify-center text-center border-l-2 border-gray-700">
                <Brain className="w-5 h-5 text-gray-400 mb-2" />
                <span className="text-xs text-gray-500 uppercase">Architecture</span>
                <span className="text-white font-medium">{activeModelData.display_name}</span>
              </div>
              <div className="glass-card p-4 flex flex-col items-center justify-center text-center border-l-2 border-emerald-500">
                <Target className="w-5 h-5 text-emerald-400 mb-2" />
                <span className="text-xs text-gray-500 uppercase">Test Accuracy</span>
                <span className="text-emerald-400 font-bold text-lg">
                  {activeModelData.acc ? (activeModelData.acc * 100).toFixed(1) + "%" : "N/A"}
                </span>
              </div>
              <div className="glass-card p-4 flex flex-col items-center justify-center text-center border-l-2 border-sky-500">
                <Activity className="w-5 h-5 text-sky-400 mb-2" />
                <span className="text-xs text-gray-500 uppercase">Macro F1 Score</span>
                <span className="text-sky-400 font-bold text-lg">
                  {activeModelData.f1 ? (activeModelData.f1 * 100).toFixed(1) + "%" : "N/A"}
                </span>
              </div>
              <div className="glass-card p-4 flex flex-col items-center justify-center text-center border-l-2 border-purple-500">
                <LineChart className="w-5 h-5 text-purple-400 mb-2" />
                <span className="text-xs text-gray-500 uppercase">ROC-AUC</span>
                <span className="text-purple-400 font-bold text-lg">
                  {activeModelData.auc ? (activeModelData.auc * 100).toFixed(1) + "%" : "N/A"}
                </span>
              </div>
            </div>

            {/* Learning Curves */}
            <div className="glass-card p-6 md:p-8">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-xl font-bold text-white">Learning Curves</h2>
                  <p className="text-sm text-gray-400 mt-1">Training vs Validation Loss and Accuracy per Epoch</p>
                </div>
              </div>
              
              {learningCurve ? (
                <div className="relative aspect-[16/7] md:aspect-[21/7] bg-white rounded-xl overflow-hidden border border-gray-800 p-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`${API_BASE}${learningCurve.url}`}
                    alt={`${activeModelData.display_name} Learning Curves`}
                    className="w-full h-full object-contain"
                  />
                </div>
              ) : (
                <div className="h-64 bg-gray-900/50 rounded-xl flex items-center justify-center border border-gray-800">
                  <div className="text-center">
                    <Activity className="w-8 h-8 text-gray-600 mx-auto mb-2" />
                    <p className="text-gray-500 text-sm">
                      {activePhase === "advanced" && !["R1_differential_lr", "R2_mixup", "R3_warmup", "R1+R2_combined"].includes(activeModel)
                        ? "Post-hoc technique — no training curves (no retraining involved)."
                        : "Learning curves not available for this model."}
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Confusion Matrix & Per-Class Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="glass-card p-6">
                <h2 className="text-xl font-bold text-white mb-2">Confusion Matrix</h2>
                <p className="text-sm text-gray-400 mb-6">Classification results on the held-out 15% test split.</p>
                {confusionMatrix ? (
                  <div className="relative aspect-square bg-white rounded-xl overflow-hidden border border-gray-800 p-2">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`${API_BASE}${confusionMatrix.url}`}
                      alt={`${activeModelData.display_name} Confusion Matrix`}
                      className="w-full h-full object-contain"
                    />
                  </div>
                ) : (
                  <div className="aspect-square bg-gray-900/50 rounded-xl flex items-center justify-center border border-gray-800">
                    <p className="text-gray-500 text-sm">Confusion matrix unavailable.</p>
                  </div>
                )}
              </div>

              <div className="glass-card p-6">
                <h2 className="text-xl font-bold text-white mb-2">Per-Class Metrics</h2>
                <p className="text-sm text-gray-400 mb-6">Precision, Recall, and F1 per class on the test set.</p>
                {perClassBar ? (
                  <div className="relative aspect-square bg-white rounded-xl overflow-hidden border border-gray-800 p-2">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`${API_BASE}${perClassBar.url}`}
                      alt={`${activeModelData.display_name} Per-Class Metrics`}
                      className="w-full h-full object-contain"
                    />
                  </div>
                ) : (
                  <div className="aspect-square bg-gray-900/50 rounded-xl flex items-center justify-center border border-gray-800">
                    <p className="text-gray-500 text-sm">Per-class metrics unavailable.</p>
                  </div>
                )}
              </div>
            </div>

            {/* Calibration Plot (only for finetune/advanced models that have it) */}
            {calibrationPlot && (
              <div className="glass-card p-6 md:p-8">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-xl font-bold text-white flex items-center gap-2">
                      <Gauge className="w-5 h-5 text-indigo-400" />
                      Calibration (Reliability Diagram)
                    </h2>
                    <p className="text-sm text-gray-400 mt-1">Per-class confidence vs actual accuracy. Perfect calibration follows the diagonal.</p>
                  </div>
                </div>
                <div className="relative aspect-[16/5] bg-white rounded-xl overflow-hidden border border-gray-800 p-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`${API_BASE}${calibrationPlot.url}`}
                    alt={`${activeModelData.display_name} Calibration Plot`}
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function TrainingPage() {
  return (
    <AuthGuard>
      <TrainingContent />
    </AuthGuard>
  );
}
