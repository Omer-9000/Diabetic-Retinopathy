import { Activity } from "lucide-react";

export default function Footer() {
  const techStack = ["PyTorch", "Next.js", "Flask", "Grad-CAM"];

  return (
    <footer className="border-t border-white/5 bg-gray-950/50 backdrop-blur-sm mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand */}
          <div>
            <div className="flex items-center space-x-2 mb-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center">
                <Activity className="w-4 h-4 text-white" />
              </div>
              <span className="font-bold text-white">
                Retina<span className="text-sky-400">AI</span>
              </span>
            </div>
            <p className="text-sm text-gray-500 leading-relaxed max-w-xs">
              Research-grade deep learning platform for automated diabetic retinopathy detection and
              severity classification.
            </p>
          </div>

          {/* Tech Stack */}
          <div>
            <h4 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">
              Built With
            </h4>
            <div className="flex flex-wrap gap-2">
              {techStack.map((tech) => (
                <span
                  key={tech}
                  className="px-2.5 py-1 text-xs font-medium text-gray-400 bg-white/5 border border-white/10 rounded-md"
                >
                  {tech}
                </span>
              ))}
            </div>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">
              Project
            </h4>
            <div className="space-y-2">
              <p className="text-sm text-gray-500">
                AI Healthcare Research Project
              </p>
              <p className="text-sm text-gray-500">
                Diabetic Retinopathy Severity Classification
              </p>
              <p className="text-sm text-gray-500">
                Multi-Model Comparative Analysis
              </p>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-8 pt-6 border-t border-white/5 flex flex-col sm:flex-row justify-between items-center gap-3">
          <p className="text-xs text-gray-600">
            &copy; {new Date().getFullYear()} RetinaAI Research Platform. All rights reserved.
          </p>
          <p className="text-xs text-gray-600">
            For research and educational purposes only. Not a medical device.
          </p>
        </div>
      </div>
    </footer>
  );
}
