import { Network, Zap, Target, BookOpen, Layers } from "lucide-react";

export default function ResearchPage() {
  const architectures = [
    {
      name: "ResNet-50",
      year: 2015,
      origin: "Microsoft Research",
      params: "23.5M",
      flops: "4.1 GFLOPs",
      innovation: "Skip connections (residual learning)",
      description: "Introduced residual blocks that allow gradients to flow directly through skip connections, solving the vanishing gradient problem in very deep networks. Still considered a robust baseline for medical image analysis.",
      strengths: ["Highly stable training", "Excellent feature extraction", "Well-studied in medical domain"],
      weaknesses: ["Computationally heavy", "Large memory footprint"],
    },
    {
      name: "EfficientNet (B0 & B3)",
      year: 2019,
      origin: "Google Brain",
      params: "B0: 4.0M | B3: 10.7M",
      flops: "B0: 0.4 GFLOPs | B3: 1.8 GFLOPs",
      innovation: "Compound scaling method",
      description: "Instead of arbitrarily scaling depth or width, EfficientNet uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. Provides state-of-the-art accuracy with order-of-magnitude fewer parameters.",
      strengths: ["Extremely parameter efficient", "Fast inference", "Balances spatial and channel features well"],
      weaknesses: ["Requires precise hyperparameter tuning"],
    },
    {
      name: "DenseNet-121",
      year: 2017,
      origin: "Cornell University",
      params: "7.0M",
      flops: "2.9 GFLOPs",
      innovation: "Dense connectivity pattern",
      description: "Connects each layer to every other layer in a feed-forward fashion. This maximizes information and gradient flow, making it exceptionally good at detecting fine-grained patterns like microaneurysms in retinal scans.",
      strengths: ["Strong gradient flow", "Feature reuse", "Excellent on small datasets"],
      weaknesses: ["High memory usage during training due to concatenation"],
    },
    {
      name: "MobileNetV3-Large",
      year: 2019,
      origin: "Google",
      params: "4.2M",
      flops: "0.2 GFLOPs",
      innovation: "Hardware-aware NAS + h-swish",
      description: "Optimized for mobile device CPUs using Network Architecture Search. Uses inverted residuals, linear bottlenecks, and lightweight attention (Squeeze-and-Excitation) modules.",
      strengths: ["Fastest inference", "Lowest memory footprint", "Edge-device ready"],
      weaknesses: ["Lower representational capacity than heavier models"],
    },
    {
      name: "ViT-B/16 (Vision Transformer)",
      year: 2020,
      origin: "Google Brain",
      params: "85.8M",
      flops: "17.6 GFLOPs",
      innovation: "Pure self-attention on image patches",
      description: "Breaks the image into 16x16 patches and processes them with a standard Transformer encoder. Completely abandons convolutions, relying entirely on self-attention to model global context.",
      strengths: ["Captures global spatial relationships better than CNNs", "High robustness"],
      weaknesses: ["Massive parameter count", "Data hungry", "Slowest inference"],
    }
  ];

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-12">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <h1 className="text-4xl font-extrabold tracking-tight text-white mb-4">
          Neural Architecture Research
        </h1>
        <p className="text-lg text-gray-400">
          In-depth analysis of the 6 deep learning architectures evaluated for diabetic retinopathy classification.
        </p>
      </div>

      {/* Methodology Section */}
      <section className="glass-card p-8">
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
          <BookOpen className="w-6 h-6 mr-3 text-sky-400" />
          Training Methodology
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-lg font-semibold text-gray-200 mb-3">Loss Function: Focal Loss</h3>
            <p className="text-gray-400 text-sm leading-relaxed mb-4">
              Medical datasets are heavily imbalanced (e.g., 50% No DR, 8% Severe DR). Standard Cross-Entropy loss causes the model to overfit on the majority class. We implemented <strong>Focal Loss (γ=2.0)</strong> with dynamic class weighting to down-weight easy examples and focus training on hard, rare cases (Severe/Proliferative).
            </p>
            
            <h3 className="text-lg font-semibold text-gray-200 mb-3">Transfer Learning</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              All models were initialized with ImageNet pre-trained weights. We replaced the final classification head with a 5-node Linear layer. The entire network was fine-tuned using AdamW optimizer with a learning rate scheduler (ReduceLROnPlateau).
            </p>
          </div>
          
          <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800">
            <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Pipeline Specs</h3>
            <ul className="space-y-4">
              <li className="flex items-start">
                <Target className="w-5 h-5 text-sky-400 mr-3 shrink-0" />
                <div>
                  <span className="text-gray-200 block">Optimizer</span>
                  <span className="text-gray-500 text-sm">AdamW (lr=1e-4, weight_decay=1e-4)</span>
                </div>
              </li>
              <li className="flex items-start">
                <Zap className="w-5 h-5 text-sky-400 mr-3 shrink-0" />
                <div>
                  <span className="text-gray-200 block">Mixed Precision</span>
                  <span className="text-gray-500 text-sm">torch.cuda.amp.GradScaler for faster training</span>
                </div>
              </li>
              <li className="flex items-start">
                <Layers className="w-5 h-5 text-sky-400 mr-3 shrink-0" />
                <div>
                  <span className="text-gray-200 block">Early Stopping</span>
                  <span className="text-gray-500 text-sm">Patience of 7 epochs on Validation Loss</span>
                </div>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Architectures Grid */}
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-white flex items-center px-2">
          <Network className="w-6 h-6 mr-3 text-sky-400" />
          Evaluated Architectures
        </h2>
        
        <div className="grid grid-cols-1 gap-6">
          {architectures.map((arch, idx) => (
            <div 
              key={arch.name}
              className="glass-card p-6 md:p-8"
            >
              <div className="flex flex-col md:flex-row md:items-start justify-between gap-6">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="text-2xl font-bold text-white">{arch.name}</h3>
                    <span className="px-2.5 py-1 bg-gray-800 text-gray-300 text-xs rounded-md border border-gray-700">
                      {arch.year}
                    </span>
                  </div>
                  <p className="text-sky-400 text-sm font-medium mb-4">{arch.origin} • Innovation: {arch.innovation}</p>
                  
                  <p className="text-gray-400 text-sm leading-relaxed mb-6">
                    {arch.description}
                  </p>
                  
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <h4 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-2">Strengths</h4>
                      <ul className="list-disc list-inside text-sm text-gray-400 space-y-1">
                        {arch.strengths.map(s => <li key={s}>{s}</li>)}
                      </ul>
                    </div>
                    <div>
                      <h4 className="text-xs font-semibold text-rose-400 uppercase tracking-wider mb-2">Weaknesses</h4>
                      <ul className="list-disc list-inside text-sm text-gray-400 space-y-1">
                        {arch.weaknesses.map(w => <li key={w}>{w}</li>)}
                      </ul>
                    </div>
                  </div>
                </div>
                
                <div className="w-full md:w-64 shrink-0 bg-gray-900/80 p-5 rounded-xl border border-gray-800 h-fit">
                  <h4 className="text-sm font-medium text-gray-300 mb-4 border-b border-gray-800 pb-2">Technical Specs</h4>
                  <div className="space-y-4">
                    <div>
                      <span className="block text-xs text-gray-500 mb-1">Parameters</span>
                      <span className="font-mono text-white">{arch.params}</span>
                    </div>
                    <div>
                      <span className="block text-xs text-gray-500 mb-1">Computational Cost</span>
                      <span className="font-mono text-white">{arch.flops}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
