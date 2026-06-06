"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Clock, ShieldCheck, AlertTriangle, Trash2, Cpu, FileDown, Activity } from "lucide-react";
import AuthGuard from "@/components/AuthGuard";
import { API_BASE, authFetch } from "@/lib/auth";

interface HistoryItem {
  id: string;
  date: string;
  image: string;
  score?: number;
  confidence?: number;
  threshold?: number;
  is_diabetic: boolean;
  message: string;
  severity_label?: string;
  predicted_class_display?: string;
  model_name?: string;
  display_name?: string;
  grad_cam?: string;
}

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [filter, setFilter] = useState<string>("all");
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await authFetch(`${API_BASE}/history`);
        if (!response.ok) {
          throw new Error("Failed to fetch history");
        }
        const data = await response.json();
        const mappedHistory: HistoryItem[] = (data.history || []).map((item: any) => {
          let formattedDate = item.timestamp;
          try {
            if (item.timestamp) {
              formattedDate = new Date(item.timestamp).toLocaleString();
            }
          } catch (e) {
            console.error("Failed to parse date", item.timestamp);
          }

          return {
            id: item._id,
            date: formattedDate,
            image: item.original_image,
            grad_cam: item.gradcam_image,
            confidence: item.confidence_score,
            is_diabetic: item.is_diabetic,
            predicted_class_display: item.predicted_class,
            display_name: item.model_used,
            model_name: item.model_name,
            message: item.predicted_class,
          };
        });
        setHistory(mappedHistory);
      } catch (err: any) {
        console.error(err);
        setError("Could not load your history. Please try again.");
      } finally {
        setIsLoadingHistory(false);
      }
    };
    fetchHistory();
  }, []);

  const clearHistory = async () => {
    if (confirm("Are you sure you want to clear all patient history? This cannot be undone.")) {
      try {
        const response = await authFetch(`${API_BASE}/history`, {
          method: "DELETE",
        });
        if (!response.ok) {
          throw new Error("Failed to clear history");
        }
        setHistory([]);
      } catch (err: any) {
        alert(err.message || "Failed to clear history");
      }
    }
  };
  
  const exportCSV = () => {
    if (history.length === 0) return;
    
    const headers = ["ID", "Date", "Model Used", "Prediction", "Confidence", "Is Diabetic"];
    const rows = history.map(item => [
      item.id,
      `"${item.date}"`,
      `"${item.display_name || item.model_name || 'Unknown'}"`,
      `"${item.predicted_class_display || item.severity_label || 'Unknown'}"`,
      (item.confidence || item.score || 0).toFixed(4),
      item.is_diabetic ? 'Yes' : 'No'
    ]);
    
    const csvContent = "data:text/csv;charset=utf-8," 
      + headers.join(",") + "\n" 
      + rows.map(e => e.join(",")).join("\n");
      
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `retina_history_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredHistory = history.filter(item => {
    if (filter === "all") return true;
    if (filter === "dr") return item.is_diabetic;
    if (filter === "nodr") return !item.is_diabetic;
    return true;
  });

  return (
    <AuthGuard>
      <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-8">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-4xl font-extrabold tracking-tight text-white mb-2">
              Patient History
            </h1>
            <p className="text-gray-400">Past retinal scans and predictions (Stored securely in database)</p>
          </div>
        
        {history.length > 0 && (
          <div className="flex items-center space-x-3">
             <div className="bg-gray-900/50 p-1 rounded-lg border border-gray-800 flex space-x-1">
               <button onClick={() => setFilter("all")} className={`px-3 py-1.5 text-xs font-medium rounded-md ${filter === 'all' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white'}`}>All</button>
               <button onClick={() => setFilter("dr")} className={`px-3 py-1.5 text-xs font-medium rounded-md ${filter === 'dr' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white'}`}>DR Only</button>
               <button onClick={() => setFilter("nodr")} className={`px-3 py-1.5 text-xs font-medium rounded-md ${filter === 'nodr' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white'}`}>Healthy Only</button>
             </div>
             
            <button onClick={exportCSV} className="p-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-colors border border-gray-700" title="Export CSV">
              <FileDown className="w-5 h-5" />
            </button>
             
            <button 
              onClick={clearHistory}
              className="p-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors border border-red-500/30"
              title="Clear History"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          </div>
        )}
      </div>

      {isLoadingHistory ? (
        <div className="text-center py-32 glass-card flex flex-col items-center justify-center">
          <Activity className="w-12 h-12 text-sky-400 animate-spin mb-4" />
          <p className="text-gray-400">Loading history records...</p>
        </div>
      ) : error ? (
        <div className="text-center py-32 glass-card">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-6" />
          <h3 className="text-2xl font-medium text-red-400">Error Loading History</h3>
          <p className="text-gray-500 mt-2 max-w-sm mx-auto">{error}</p>
        </div>
      ) : history.length === 0 ? (
        <div className="text-center py-32 glass-card">
          <Clock className="w-16 h-16 text-gray-700 mx-auto mb-6" />
          <h3 className="text-2xl font-medium text-gray-300">No History Found</h3>
          <p className="text-gray-500 mt-2 max-w-sm mx-auto">Upload and analyze a retinal scan on the Predict page to see it recorded here.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredHistory.map((item, index) => (
            <motion.div 
              key={item.id}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              className="glass-card overflow-hidden group"
            >
              <div className="aspect-video bg-black relative border-b border-gray-800 overflow-hidden flex">
                <div className="flex-1 relative border-r border-gray-800/50">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={item.image} alt={`Scan from ${item.date}`} className="absolute inset-0 w-full h-full object-contain p-2" />
                  <span className="absolute bottom-1 left-2 text-[10px] text-gray-500">Original</span>
                </div>
                {item.grad_cam ? (
                  <div className="flex-1 relative bg-black/80">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={item.grad_cam} alt="Grad-CAM" className="absolute inset-0 w-full h-full object-contain mix-blend-screen p-2" />
                    <span className="absolute bottom-1 right-2 text-[10px] text-gray-500">Grad-CAM</span>
                  </div>
                ) : (
                  <div className="flex-1 flex items-center justify-center bg-gray-950">
                     <span className="text-xs text-gray-700">No CAM</span>
                  </div>
                )}
                
                <div className="absolute top-2 left-2 bg-black/70 backdrop-blur-md px-2.5 py-1 rounded-md text-[10px] font-medium text-gray-300 border border-gray-700">
                  {item.date}
                </div>
              </div>
              
              <div className={`p-5 ${item.is_diabetic ? 'bg-red-950/20' : 'bg-emerald-950/20'}`}>
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    {item.is_diabetic ? (
                      <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                    ) : (
                      <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                    )}
                    <div>
                      <h3 className={`font-bold leading-tight ${item.is_diabetic ? 'text-red-400' : 'text-emerald-400'}`}>
                        {item.predicted_class_display || item.severity_label}
                      </h3>
                      <p className="text-[10px] text-gray-500 mt-1 flex items-center">
                        <Cpu className="w-3 h-3 mr-1" />
                        {item.display_name || item.model_name || "Legacy Model"}
                      </p>
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <span className="text-[10px] text-gray-500 uppercase block mb-0.5">Confidence</span>
                    <span className="font-mono font-bold text-white text-lg">
                      {((item.confidence || item.score || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
          
          {filteredHistory.length === 0 && (
            <div className="col-span-full text-center py-12">
               <p className="text-gray-500">No records match the current filter.</p>
            </div>
          )}
        </div>
      )}
      </div>
    </AuthGuard>
  );
}
