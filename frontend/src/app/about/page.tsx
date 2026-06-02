"use client";

import { Heart, Shield, Activity, Cpu } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 max-w-4xl mx-auto space-y-12">
      {/* Header */}
      <div className="text-center mb-16">
        <div className="w-20 h-20 bg-gradient-to-br from-sky-500 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg shadow-sky-500/20">
          <Activity className="w-10 h-10 text-white" />
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl mb-4">
          About RetinaAI
        </h1>
        <p className="text-xl text-gray-400">
          A research initiative applying state-of-the-art computer vision to ophthalmic diagnostics.
        </p>
      </div>

      {/* Overview */}
      <div className="glass-card p-8">
        <h2 className="text-2xl font-bold text-white mb-4">Project Motivation</h2>
        <div className="prose prose-invert max-w-none text-gray-300">
          <p>
            Diabetic retinopathy is a leading cause of blindness worldwide, affecting nearly a third of all people with diabetes. Regular retinal screening is critical, as early detection and treatment can prevent vision loss in up to 90% of cases. 
          </p>
          <p>
            However, manual grading of retinal fundus images is time-consuming, expensive, and subject to human error/fatigue. There is a massive global shortage of trained ophthalmologists, especially in developing regions.
          </p>
          <p>
            This project aims to bridge that gap by providing a robust, explainable, multi-model automated screening platform.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-card p-6">
          <Shield className="w-8 h-8 text-emerald-400 mb-4" />
          <h3 className="text-lg font-bold text-white mb-2">Clinical Safety & Explainability</h3>
          <p className="text-sm text-gray-400 leading-relaxed">
            Medical AI cannot be a black box. We implemented Grad-CAM across all CNN and Transformer architectures to highlight the exact lesions, exudates, and microaneurysms that influence the prediction, building trust with clinicians.
          </p>
        </div>
        
        <div className="glass-card p-6">
          <Cpu className="w-8 h-8 text-sky-400 mb-4" />
          <h3 className="text-lg font-bold text-white mb-2">Architectural Rigor</h3>
          <p className="text-sm text-gray-400 leading-relaxed">
            Rather than relying on a single architecture, this platform evaluates 6 distinct structural paradigms (Residual, Dense, Compound Scaling, Mobile/Edge, and Self-Attention/Transformers) to find the optimal balance of accuracy and computational cost.
          </p>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="glass-card p-8">
        <h2 className="text-2xl font-bold text-white mb-6">Technology Stack</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-900/80 p-4 rounded-xl border border-gray-800 text-center">
            <span className="block text-xs text-gray-500 uppercase tracking-wider mb-2">Deep Learning</span>
            <span className="font-bold text-white">PyTorch</span>
          </div>
          <div className="bg-gray-900/80 p-4 rounded-xl border border-gray-800 text-center">
            <span className="block text-xs text-gray-500 uppercase tracking-wider mb-2">Backend API</span>
            <span className="font-bold text-white">Flask (Python)</span>
          </div>
          <div className="bg-gray-900/80 p-4 rounded-xl border border-gray-800 text-center">
            <span className="block text-xs text-gray-500 uppercase tracking-wider mb-2">Frontend Web</span>
            <span className="font-bold text-white">Next.js 16 (React)</span>
          </div>
          <div className="bg-gray-900/80 p-4 rounded-xl border border-gray-800 text-center">
            <span className="block text-xs text-gray-500 uppercase tracking-wider mb-2">Styling</span>
            <span className="font-bold text-white">Tailwind CSS v4</span>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-2xl p-6 flex items-start">
        <Heart className="w-6 h-6 text-amber-500 mr-4 shrink-0 mt-0.5" />
        <div>
          <h3 className="text-lg font-bold text-amber-400 mb-2">Medical Disclaimer</h3>
          <p className="text-sm text-amber-500/80 leading-relaxed">
            This software is a research prototype intended for educational and research purposes only. It is not an FDA-approved medical device and must not be used for primary diagnostic purposes. Always consult a qualified ophthalmologist or healthcare provider for medical diagnosis and treatment.
          </p>
        </div>
      </div>
    </div>
  );
}
