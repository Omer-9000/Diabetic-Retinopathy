"use client";

import { useState } from "react";
import { Eye, EyeOff, Maximize2, SlidersHorizontal } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface GradCAMViewerProps {
  originalImage: string;
  gradCamImage: string | null;
  modelName?: string;
}

export default function GradCAMViewer({
  originalImage,
  gradCamImage,
  modelName,
}: GradCAMViewerProps) {
  const [opacity, setOpacity] = useState<number>(0.7);
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  if (!gradCamImage) {
    return (
      <div className="w-full aspect-square bg-gray-900 rounded-2xl flex items-center justify-center border border-gray-800">
        <p className="text-sm text-gray-500">Grad-CAM not available for this model.</p>
      </div>
    );
  }

  const ImageContent = () => (
    <div className="relative w-full aspect-square bg-black overflow-hidden rounded-xl border border-gray-800 group">
      {/* Original Image */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={originalImage}
        alt="Original Retinal Scan"
        className="absolute inset-0 w-full h-full object-contain"
      />

      {/* Grad-CAM Overlay */}
      <AnimatePresence>
        {showOverlay && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="absolute inset-0 w-full h-full"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={gradCamImage}
              alt="Grad-CAM Heatmap"
              className="w-full h-full object-contain mix-blend-screen"
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Expand button on hover (only in non-expanded mode) */}
      {!isExpanded && (
        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
          <button
            onClick={() => setIsExpanded(true)}
            className="p-3 bg-white/10 backdrop-blur-md rounded-full pointer-events-auto hover:bg-white/20 transition-colors"
          >
            <Maximize2 className="w-6 h-6 text-white" />
          </button>
        </div>
      )}
    </div>
  );

  return (
    <>
      <div className="w-full space-y-4">
        {/* Main Viewer */}
        <div className="relative">
          <ImageContent />
        </div>

        {/* Controls */}
        <div className="bg-gray-900/50 p-4 rounded-xl border border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-sm font-semibold text-gray-300 flex items-center">
              <Eye className="w-4 h-4 mr-2 text-sky-400" />
              Grad-CAM Explainability
              {modelName && <span className="ml-2 text-xs text-gray-500 font-normal">({modelName})</span>}
            </h4>
            <button
              onClick={() => setShowOverlay(!showOverlay)}
              className="p-1.5 rounded-md hover:bg-gray-800 text-gray-400 hover:text-white transition-colors"
              title={showOverlay ? "Hide Overlay" : "Show Overlay"}
            >
              {showOverlay ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>

          <div className="flex items-center space-x-3">
            <SlidersHorizontal className="w-4 h-4 text-gray-500" />
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              disabled={!showOverlay}
              className={`w-full h-1.5 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-sky-500 ${
                !showOverlay ? "opacity-50" : ""
              }`}
            />
            <span className="text-xs text-gray-500 w-8 text-right font-mono">
              {Math.round(opacity * 100)}%
            </span>
          </div>
          <p className="text-[10px] text-gray-500 mt-3 text-center">
            Heatmap highlights regions influencing the model's prediction. Red/yellow areas indicate higher importance.
          </p>
        </div>
      </div>

      {/* Expanded Modal */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-black/95 backdrop-blur-xl flex items-center justify-center p-4 sm:p-8"
          >
            <div className="relative w-full max-w-5xl h-full flex flex-col">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-white">Grad-CAM Visualization {modelName ? `— ${modelName}` : ''}</h3>
                <button
                  onClick={() => setIsExpanded(false)}
                  className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
                >
                  Close
                </button>
              </div>
              
              <div className="flex-1 min-h-0 flex flex-col md:flex-row gap-6">
                <div className="flex-1 flex flex-col items-center justify-center">
                  <span className="text-sm text-gray-400 mb-2">Original</span>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={originalImage} alt="Original" className="max-w-full max-h-[80vh] object-contain rounded-xl border border-gray-800" />
                </div>
                <div className="flex-1 flex flex-col items-center justify-center">
                   <span className="text-sm text-gray-400 mb-2">Grad-CAM</span>
                   {/* eslint-disable-next-line @next/next/no-img-element */}
                   <img src={gradCamImage} alt="Grad-CAM" className="max-w-full max-h-[80vh] object-contain rounded-xl border border-sky-500/30" />
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
