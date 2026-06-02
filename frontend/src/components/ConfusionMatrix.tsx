"use client";

import { useState } from "react";
import { Maximize2, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface ConfusionMatrixProps {
  modelName: string;
  type: "val" | "test";
  url: string;
}

export default function ConfusionMatrix({ modelName, type, url }: ConfusionMatrixProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <>
      <div className="relative group rounded-xl overflow-hidden border border-gray-800 bg-gray-900/50">
        <div className="p-2 border-b border-gray-800 bg-gray-900/80">
          <p className="text-xs text-center text-gray-400 font-medium truncate">
            {modelName} ({type === "test" ? "Test Set" : "Validation Set"})
          </p>
        </div>
        
        <div className="relative aspect-square cursor-pointer" onClick={() => setIsExpanded(true)}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`http://localhost:5000${url}`}
            alt={`${modelName} Confusion Matrix`}
            className="w-full h-full object-contain p-2"
          />
          
          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
            <div className="p-2 bg-gray-900/80 backdrop-blur-sm rounded-lg text-white">
              <Maximize2 className="w-5 h-5" />
            </div>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex items-center justify-center p-4 sm:p-8"
            onClick={() => setIsExpanded(false)}
          >
            <div 
              className="relative w-full max-w-4xl max-h-full flex flex-col items-center justify-center bg-gray-950 p-6 rounded-2xl border border-gray-800"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="w-full flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium text-white">
                  Confusion Matrix: {modelName}
                </h3>
                <button
                  onClick={() => setIsExpanded(false)}
                  className="p-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-full transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="relative w-full h-[70vh] flex items-center justify-center bg-black/50 rounded-xl">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`http://localhost:5000${url}`}
                  alt={`${modelName} Confusion Matrix (Expanded)`}
                  className="max-w-full max-h-full object-contain"
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
