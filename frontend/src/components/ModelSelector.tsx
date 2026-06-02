"use client";

import { motion } from "framer-motion";
import { Cpu, Zap, Activity } from "lucide-react";

export type ModelData = {
  name: string;
  display_name: string;
  family: string;
  year: number;
  origin: string;
  key_innovation: string;
  complexity: string;
  flops: string;
  available: boolean;
  is_best: boolean;
  accuracy: number;
  f1_score: number;
  precision: number;
  recall: number;
  roc_auc: number;
  training_time_s: number;
  num_parameters: number;
};

interface ModelSelectorProps {
  models: ModelData[];
  selectedModel: string;
  onSelectModel: (model: string) => void;
  isLoading: boolean;
}

export default function ModelSelector({
  models,
  selectedModel,
  onSelectModel,
  isLoading,
}: ModelSelectorProps) {
  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Select AI Model</h3>
        <span className="text-xs font-medium px-2.5 py-1 bg-sky-500/10 text-sky-400 border border-sky-500/20 rounded-full">
          {models.length} Models Available
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {/* Compare All Option */}
        <button
          disabled={isLoading}
          onClick={() => onSelectModel("compare_all")}
          className={`relative p-4 rounded-xl border text-left transition-all ${
            selectedModel === "compare_all"
              ? "bg-gradient-to-br from-sky-500/10 to-indigo-600/10 border-sky-500 shadow-lg shadow-sky-500/20"
              : "bg-gray-950/50 border-gray-800 hover:border-gray-600 hover:bg-gray-900/50"
          } ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          {selectedModel === "compare_all" && (
            <div className="absolute top-0 right-0 w-12 h-12 overflow-hidden rounded-tr-xl">
              <div className="absolute top-[-10px] right-[-10px] w-6 h-6 bg-sky-500 rounded-full blur-xl opacity-50" />
            </div>
          )}
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center space-x-2">
              <div
                className={`p-1.5 rounded-lg ${
                  selectedModel === "compare_all"
                    ? "bg-sky-500/20 text-sky-400"
                    : "bg-gray-800 text-gray-400"
                }`}
              >
                <Activity className="w-4 h-4" />
              </div>
              <span
                className={`font-semibold ${
                  selectedModel === "compare_all" ? "text-sky-400" : "text-gray-300"
                }`}
              >
                Compare All Models
              </span>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-1 line-clamp-2">
            Run inference sequentially through all {models.length} models to get a majority vote and comparative analysis.
          </p>
        </button>

        {/* Individual Models */}
        {models.map((model) => (
          <button
            key={model.name}
            disabled={!model.available || isLoading}
            onClick={() => onSelectModel(model.name)}
            className={`relative p-4 rounded-xl border text-left transition-all ${
              !model.available
                ? "bg-gray-950/30 border-gray-900 opacity-50 cursor-not-allowed"
                : selectedModel === model.name
                ? "bg-gradient-to-br from-teal-500/10 to-emerald-600/10 border-teal-500 shadow-lg shadow-teal-500/20"
                : "bg-gray-950/50 border-gray-800 hover:border-gray-600 hover:bg-gray-900/50"
            }`}
          >
            {model.is_best && (
              <span className="absolute -top-2.5 -right-2.5 px-2 py-0.5 bg-amber-500 text-white text-[10px] font-bold uppercase rounded-full shadow-lg shadow-amber-500/20 z-10">
                Best
              </span>
            )}
            
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center space-x-2">
                <div
                  className={`p-1.5 rounded-lg ${
                    selectedModel === model.name
                      ? "bg-teal-500/20 text-teal-400"
                      : "bg-gray-800 text-gray-400"
                  }`}
                >
                  <Cpu className="w-4 h-4" />
                </div>
                <span
                  className={`font-semibold ${
                    selectedModel === model.name ? "text-white" : "text-gray-300"
                  }`}
                >
                  {model.display_name}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 mt-3">
              <div className="flex items-center space-x-1 text-xs text-gray-500">
                <Activity className="w-3.5 h-3.5" />
                <span>
                  Acc: {model.accuracy ? (model.accuracy * 100).toFixed(1) + "%" : "N/A"}
                </span>
              </div>
              <div className="flex items-center space-x-1 text-xs text-gray-500">
                <Zap className="w-3.5 h-3.5" />
                <span>
                  {(model.num_parameters / 1000000).toFixed(1)}M Params
                </span>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
