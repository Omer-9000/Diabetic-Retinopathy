"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, LineChart, Brain, Target, ArrowRight } from "lucide-react";
import ConfusionMatrix from "@/components/ConfusionMatrix";

export default function TrainingPage() {
  const [plots, setPlots] = useState<any[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeModel, setActiveModel] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("http://localhost:5000/api/models").then((res) => res.json()),
      fetch("http://localhost:5000/api/dashboard/plots").then((res) => res.json()),
    ])
      .then(([modelsData, plotsData]) => {
        setModels(modelsData.models || []);
        setPlots(plotsData.plots || []);
        if (modelsData.models && modelsData.models.length > 0) {
          // Find best model or default to first
          const best = modelsData.best_model || modelsData.models[0].name;
          setActiveModel(best);
        }
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

  const modelPlots = plots.filter((p) => p.model === activeModel);
  const learningCurve = modelPlots.find((p) => p.type === "learning_curves");
  const valMatrix = modelPlots.find((p) => p.type === "confusion_matrix");
  const testMatrix = modelPlots.find((p) => p.type === "test_confusion_matrix");
  const activeModelData = models.find((m) => m.name === activeModel);

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
          Detailed learning curves and confusion matrices captured during the model fine-tuning phase.
        </p>
      </div>

      {/* Model Selector Tabs */}
      <div className="flex flex-wrap justify-center gap-2 mb-8">
        {models.map((model) => (
          <button
            key={model.name}
            onClick={() => setActiveModel(model.name)}
            className={`px-5 py-2.5 rounded-xl text-sm font-medium transition-all ${
              activeModel === model.name
                ? "bg-gradient-to-r from-sky-500/20 to-indigo-500/20 text-sky-400 border border-sky-500/50 shadow-lg shadow-sky-500/10"
                : "bg-gray-900/50 text-gray-400 border border-gray-800 hover:bg-gray-800 hover:text-white"
            }`}
          >
            {model.display_name}
            {model.is_best && (
              <span className="ml-2 px-1.5 py-0.5 text-[10px] uppercase bg-amber-500/20 text-amber-400 rounded border border-amber-500/30">
                Best
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
              <div className="glass-card p-4 flex flex-col items-center justify-center text-center">
                <Brain className="w-5 h-5 text-gray-400 mb-2" />
                <span className="text-xs text-gray-500 uppercase">Architecture</span>
                <span className="text-white font-medium">{activeModelData.display_name}</span>
              </div>
              <div className="glass-card p-4 flex flex-col items-center justify-center text-center">
                <Target className="w-5 h-5 text-emerald-400 mb-2" />
                <span className="text-xs text-gray-500 uppercase">Test Accuracy</span>
                <span className="text-emerald-400 font-bold text-lg">
                  {activeModelData.accuracy ? (activeModelData.accuracy * 100).toFixed(1) + "%" : "N/A"}
                </span>
              </div>
              <div className="glass-card p-4 flex flex-col items-center justify-center text-center">
                <Activity className="w-5 h-5 text-sky-400 mb-2" />
                <span className="text-xs text-gray-500 uppercase">F1 Score</span>
                <span className="text-sky-400 font-bold text-lg">
                  {activeModelData.f1_score ? (activeModelData.f1_score * 100).toFixed(1) + "%" : "N/A"}
                </span>
              </div>
              <div className="glass-card p-4 flex flex-col items-center justify-center text-center">
                <LineChart className="w-5 h-5 text-purple-400 mb-2" />
                <span className="text-xs text-gray-500 uppercase">ROC-AUC</span>
                <span className="text-purple-400 font-bold text-lg">
                  {activeModelData.roc_auc ? (activeModelData.roc_auc * 100).toFixed(1) + "%" : "N/A"}
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
                    src={`http://localhost:5000${learningCurve.url}`}
                    alt={`${activeModelData.display_name} Learning Curves`}
                    className="w-full h-full object-contain"
                  />
                </div>
              ) : (
                <div className="h-64 bg-gray-900/50 rounded-xl flex items-center justify-center border border-gray-800">
                  <p className="text-gray-500">Learning curves not available for this model.</p>
                </div>
              )}
            </div>

            {/* Confusion Matrices */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="glass-card p-6">
                <h2 className="text-xl font-bold text-white mb-2">Validation Phase</h2>
                <p className="text-sm text-gray-400 mb-6">Confusion matrix generated during model evaluation on the 15% validation split.</p>
                {valMatrix ? (
                  <ConfusionMatrix
                    modelName={activeModelData.display_name}
                    type="val"
                    url={valMatrix.url}
                  />
                ) : (
                  <div className="aspect-square bg-gray-900/50 rounded-xl flex items-center justify-center border border-gray-800">
                    <p className="text-gray-500">Validation matrix unavailable.</p>
                  </div>
                )}
              </div>

              <div className="glass-card p-6">
                <h2 className="text-xl font-bold text-white mb-2">Testing Phase</h2>
                <p className="text-sm text-gray-400 mb-6">Confusion matrix generated during final inference on the held-out 15% test split.</p>
                {testMatrix ? (
                  <ConfusionMatrix
                    modelName={activeModelData.display_name}
                    type="test"
                    url={testMatrix.url}
                  />
                ) : (
                  <div className="aspect-square bg-gray-900/50 rounded-xl flex items-center justify-center border border-gray-800">
                    <p className="text-gray-500">Test matrix unavailable.</p>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
