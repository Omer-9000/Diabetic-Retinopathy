"use client";

import React, { useState, useRef, useEffect } from "react";
import { UploadCloud, X, Activity, AlertTriangle, ShieldCheck, Printer, RefreshCw, Layers } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ModelSelector, { ModelData } from "@/components/ModelSelector";
import ProbabilityChart from "@/components/ProbabilityChart";
import GradCAMViewer from "@/components/GradCAMViewer";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [models, setModels] = useState<ModelData[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [isCompareAll, setIsCompareAll] = useState(false);

  useEffect(() => {
    // Fetch available models from backend
    fetch("http://localhost:5000/api/models")
      .then((res) => res.json())
      .then((data) => {
        setModels(data.models);
        setSelectedModel(data.best_model);
      })
      .catch((err) => {
        console.error("Failed to load models:", err);
        setError("Failed to connect to backend server. Is it running?");
      });
  }, []);

  const handleFileChange = (file: File | null) => {
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => setIsDragging(false);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleModelSelect = (modelId: string) => {
    if (modelId === "compare_all") {
      setIsCompareAll(true);
      setSelectedModel("compare_all");
    } else {
      setIsCompareAll(false);
      setSelectedModel(modelId);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || !selectedModel) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    
    if (!isCompareAll) {
      formData.append("model", selectedModel);
    }

    const endpoint = isCompareAll ? "predict/compare" : "predict";

    try {
      const response = await fetch(`http://localhost:5000/${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Analysis failed. Please try again.");
      }

      const data = await response.json();
      if (data.error) throw new Error(data.error);
      
      setResult(data);

      // Save to history (only for single model right now)
      if (!isCompareAll) {
        saveToHistory(data);
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  const saveToHistory = (data: any) => {
    const reader = new FileReader();
    reader.readAsDataURL(selectedFile!);
    reader.onloadend = () => {
      const base64data = reader.result;
      const historyItem = {
        id: Date.now().toString(),
        date: new Date().toLocaleString(),
        image: base64data,
        ...data,
      };
      const existingHistory = JSON.parse(localStorage.getItem("retina_history") || "[]");
      localStorage.setItem("retina_history", JSON.stringify([historyItem, ...existingHistory].slice(0, 50)));
    };
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white mb-4">
          Automated Retinal Screening
        </h1>
        <p className="max-w-2xl mx-auto text-lg text-gray-400">
          Upload a fundus image for instant diabetic retinopathy severity classification using state-of-the-art deep learning architectures.
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Upload & Model Selection */}
        <div className="lg:col-span-5 space-y-6">
          {/* Upload Card */}
          <div className="glass-card p-6">
            <AnimatePresence mode="wait">
              {!selectedFile ? (
                <motion.div
                  key="upload-zone"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className={`relative border-2 border-dashed rounded-xl p-10 transition-colors flex flex-col items-center justify-center text-center cursor-pointer min-h-[300px] ${
                    isDragging
                      ? "border-sky-400 bg-sky-500/10"
                      : "border-gray-700 hover:border-gray-500 bg-gray-900/30"
                  }`}
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={onDragOver}
                  onDragLeave={onDragLeave}
                  onDrop={onDrop}
                >
                  <input
                    type="file"
                    className="hidden"
                    accept="image/*"
                    ref={fileInputRef}
                    onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
                  />
                  <div className="p-4 bg-gray-800 rounded-full mb-4">
                    <UploadCloud className="w-8 h-8 text-sky-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    Click or drag scan here
                  </h3>
                  <p className="text-gray-500 text-sm">
                    High-resolution fundus images (JPEG/PNG)
                  </p>
                </motion.div>
              ) : (
                <motion.div
                  key="preview-zone"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center"
                >
                  <div className="relative w-full aspect-square rounded-xl overflow-hidden border border-gray-700 bg-black/50 mb-6 group">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={previewUrl!}
                      alt="Preview"
                      className="w-full h-full object-contain"
                    />
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <button
                        onClick={clearSelection}
                        className="flex items-center space-x-2 px-4 py-2 bg-red-500/80 hover:bg-red-500 text-white rounded-lg backdrop-blur-md transition"
                      >
                        <X className="w-4 h-4" />
                        <span>Remove</span>
                      </button>
                    </div>
                  </div>

                  {!result && (
                    <button
                      onClick={handleUpload}
                      disabled={isLoading}
                      className="w-full relative overflow-hidden group px-6 py-4 rounded-xl font-semibold text-white bg-gradient-to-r from-sky-500 to-indigo-600 disabled:opacity-70 disabled:cursor-not-allowed transition-all"
                    >
                      <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform" />
                      <div className="relative flex items-center justify-center">
                        {isLoading ? (
                          <>
                            <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                            {isCompareAll ? "Running 6 Models..." : "Analyzing Scan..."}
                          </>
                        ) : (
                          <>
                            <Activity className="w-5 h-5 mr-2" />
                            {isCompareAll ? "Run Comparative Analysis" : "Analyze Scan"}
                          </>
                        )}
                      </div>
                    </button>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
            
            {error && (
              <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 flex items-start">
                <AlertTriangle className="w-5 h-5 mr-3 shrink-0 mt-0.5" />
                <p className="text-sm">{error}</p>
              </div>
            )}
          </div>

          {/* Model Selector Card */}
          <div className="glass-card p-6">
            <ModelSelector 
              models={models} 
              selectedModel={selectedModel} 
              onSelectModel={handleModelSelect}
              isLoading={isLoading} 
            />
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="lg:col-span-7">
          <AnimatePresence mode="wait">
            {!result && !isLoading && (
              <motion.div
                key="empty-state"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-full min-h-[400px] glass-card flex flex-col items-center justify-center p-8 text-center"
              >
                <div className="w-20 h-20 bg-gray-900 rounded-full flex items-center justify-center mb-6">
                  <Layers className="w-10 h-10 text-gray-700" />
                </div>
                <h3 className="text-xl font-medium text-gray-300 mb-2">Awaiting Image</h3>
                <p className="text-gray-500 max-w-sm">
                  Upload a retinal scan and select a model to see the severity classification and Grad-CAM explainability heatmap.
                </p>
              </motion.div>
            )}

            {isLoading && (
              <motion.div
                key="loading-state"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-full min-h-[500px] glass-card flex flex-col items-center justify-center p-8"
              >
                <div className="relative w-24 h-24 mb-8">
                  <div className="absolute inset-0 border-4 border-sky-500/20 rounded-full"></div>
                  <div className="absolute inset-0 border-4 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Activity className="w-8 h-8 text-sky-400 animate-pulse" />
                  </div>
                </div>
                <h3 className="text-xl font-medium text-white mb-2">
                  {isCompareAll ? "Running Multi-Model Ensemble" : "Processing with Deep Learning"}
                </h3>
                <p className="text-gray-400 text-center max-w-md">
                  {isCompareAll 
                    ? "Loading and running inference across all 6 architectures sequentially. This may take a minute..." 
                    : "Generating prediction probabilities and Grad-CAM attention heatmap..."}
                </p>
              </motion.div>
            )}

            {result && !isCompareAll && (
              <motion.div
                key="single-result"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-6"
              >
                {/* Main Prediction Banner */}
                <div className={`p-6 rounded-2xl border ${
                  result.is_diabetic 
                    ? 'bg-red-500/10 border-red-500/30 glow-red' 
                    : 'bg-emerald-500/10 border-emerald-500/30 glow-emerald'
                }`}>
                  <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
                    <div>
                      <div className="flex items-center space-x-3 mb-2">
                        {result.is_diabetic ? (
                          <div className="p-2 bg-red-500/20 rounded-lg">
                            <AlertTriangle className="w-6 h-6 text-red-500" />
                          </div>
                        ) : (
                          <div className="p-2 bg-emerald-500/20 rounded-lg">
                            <ShieldCheck className="w-6 h-6 text-emerald-500" />
                          </div>
                        )}
                        <h2 className="text-2xl font-bold text-white">
                          {result.message}
                        </h2>
                      </div>
                      <p className="text-gray-400 text-sm">
                        Prediction via <span className="font-semibold text-white">{result.display_name}</span>
                      </p>
                    </div>
                    
                    <div className="text-left sm:text-right bg-black/20 p-3 rounded-xl border border-white/5">
                      <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Confidence</p>
                      <p className="text-3xl font-mono font-bold text-white">
                        {(result.confidence * 100).toFixed(1)}<span className="text-lg text-gray-500">%</span>
                      </p>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Grad-CAM Viewer */}
                  <div className="glass-card p-5">
                    <GradCAMViewer 
                      originalImage={result.original_image} 
                      gradCamImage={result.grad_cam} 
                      modelName={result.display_name}
                    />
                  </div>

                  {/* Probabilities & Info */}
                  <div className="space-y-6">
                    <div className="glass-card p-5">
                      <h4 className="text-sm font-semibold text-gray-300 mb-4">Class Probabilities</h4>
                      <ProbabilityChart probabilities={result.probabilities_display} />
                    </div>

                    <div className="glass-card p-5">
                      <h4 className="text-sm font-semibold text-gray-300 mb-4">Inference Meta</h4>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center py-2 border-b border-gray-800">
                          <span className="text-sm text-gray-400">Time Taken</span>
                          <span className="text-sm font-mono text-white">{result.inference_time_ms} ms</span>
                        </div>
                        <div className="flex justify-between items-center py-2 border-b border-gray-800">
                          <span className="text-sm text-gray-400">Parameters</span>
                          <span className="text-sm font-mono text-white">
                            {result.model_params ? (result.model_params / 1000000).toFixed(1) + 'M' : 'Unknown'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center py-2">
                          <span className="text-sm text-gray-400">Test Accuracy</span>
                          <span className="text-sm font-mono text-white">
                            {result.model_accuracy ? (result.model_accuracy * 100).toFixed(1) + '%' : 'Unknown'}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex justify-end space-x-3">
                      <button onClick={clearSelection} className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-colors text-sm font-medium">
                        Analyze Another
                      </button>
                      <button onClick={() => window.print()} className="flex items-center space-x-2 px-4 py-2 bg-sky-500/20 hover:bg-sky-500/30 text-sky-400 rounded-lg transition-colors text-sm font-medium border border-sky-500/30">
                        <Printer className="w-4 h-4" />
                        <span>Print Report</span>
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {result && isCompareAll && (
              <motion.div
                key="compare-result"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-6"
              >
                {/* Ensemble Result Banner */}
                <div className={`p-6 rounded-2xl border ${
                  result.ensemble.is_diabetic 
                    ? 'bg-gradient-to-r from-red-900/40 to-orange-900/40 border-red-500/30' 
                    : 'bg-gradient-to-r from-emerald-900/40 to-teal-900/40 border-emerald-500/30'
                }`}>
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="p-2 bg-white/10 rounded-lg backdrop-blur-md">
                      <Layers className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-white">Ensemble Prediction</h2>
                      <p className="text-sm text-gray-300">Average probabilities across {result.num_models} models</p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
                    <div className="bg-black/30 p-4 rounded-xl border border-white/5">
                      <p className="text-xs text-gray-400 uppercase mb-1">Final Result</p>
                      <p className="text-xl font-bold" style={{ color: result.ensemble.severity_color }}>
                        {result.ensemble.predicted_class_display}
                      </p>
                    </div>
                    <div className="bg-black/30 p-4 rounded-xl border border-white/5">
                      <p className="text-xs text-gray-400 uppercase mb-1">Ensemble Confidence</p>
                      <p className="text-xl font-bold font-mono text-white">
                        {(result.ensemble.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-black/30 p-4 rounded-xl border border-white/5">
                      <p className="text-xs text-gray-400 uppercase mb-1">Model Agreement</p>
                      <p className="text-xl font-bold text-white">
                        {Math.round(result.majority_vote.agreement_ratio * 100)}%
                      </p>
                    </div>
                  </div>
                  
                  <div className="bg-black/20 p-4 rounded-xl border border-white/5">
                     <ProbabilityChart probabilities={result.ensemble.probabilities_display} />
                  </div>
                </div>

                {/* Individual Model Results Table */}
                <div className="glass-card overflow-hidden">
                  <div className="p-4 border-b border-gray-800 bg-gray-900/50">
                    <h3 className="text-lg font-medium text-white">Individual Model Breakdown</h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                      <thead className="text-xs text-gray-400 uppercase bg-gray-900/80 border-b border-gray-800">
                        <tr>
                          <th className="px-6 py-3">Model</th>
                          <th className="px-6 py-3">Prediction</th>
                          <th className="px-6 py-3">Confidence</th>
                          <th className="px-6 py-3 text-right">Time (ms)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.results.map((r: any, i: number) => (
                          <tr key={r.model_name} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                            <td className="px-6 py-4 font-medium text-gray-200">
                              {r.display_name}
                            </td>
                            {r.error ? (
                              <td colSpan={3} className="px-6 py-4 text-red-400 text-xs">{r.error}</td>
                            ) : (
                              <>
                                <td className="px-6 py-4">
                                  <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-gray-800 border border-gray-700" style={{ color: r.severity_color }}>
                                    {r.predicted_class_display}
                                  </span>
                                </td>
                                <td className="px-6 py-4 font-mono text-gray-300">
                                  {(r.confidence * 100).toFixed(1)}%
                                </td>
                                <td className="px-6 py-4 font-mono text-gray-400 text-right">
                                  {r.inference_time_ms}
                                </td>
                              </>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                
                {/* Grad-CAM Gallery */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  {result.results.filter((r:any) => !r.error && r.grad_cam).map((r: any) => (
                     <div key={r.model_name} className="glass-card p-2">
                       <p className="text-xs text-center text-gray-400 mb-2 truncate px-1">{r.display_name}</p>
                       <div className="relative aspect-square rounded-lg overflow-hidden bg-black">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={r.grad_cam} alt={`Grad-CAM ${r.model_name}`} className="w-full h-full object-contain" />
                       </div>
                     </div>
                  ))}
                </div>
                
                <div className="flex justify-end pt-4">
                  <button onClick={clearSelection} className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-colors text-sm font-medium">
                    Analyze New Image
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
