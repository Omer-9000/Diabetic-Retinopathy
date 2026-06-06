import { BrainCircuit, Eye, Layers, Settings2, Activity, Network } from "lucide-react";

export default function HowItWorksPage() {
  const steps = [
    {
      icon: <Eye className="w-6 h-6 text-sky-400" />,
      title: "1. The Pathology",
      description: "Diabetic retinopathy (DR) affects blood vessels in the retina. Early detection is critical. This platform automates screening by analyzing fundus images using deep Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) to detect microscopic lesions, microaneurysms, and exudates."
    },
    {
      icon: <Settings2 className="w-6 h-6 text-teal-400" />,
      title: "2. Image Preprocessing",
      description: "Before inference, the scan undergoes strict preprocessing matching the exact pipeline used during training: resizing to 224x224 pixels, conversion to a PyTorch Tensor, and normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])."
    },
    {
      icon: <BrainCircuit className="w-6 h-6 text-indigo-400" />,
      title: "3. Multi-Model Inference",
      description: "You can select from 6 state-of-the-art architectures (ResNet-50, EfficientNet B0/B3, DenseNet-121, MobileNetV3, ViT-B/16). Or, use the 'Compare All' mode which runs inference sequentially across all models and computes a majority vote ensemble."
    },
    {
      icon: <Layers className="w-6 h-6 text-purple-400" />,
      title: "4. Severity Classification",
      description: "Unlike simple binary classifiers, our models are trained to detect 5 distinct severity levels according to clinical guidelines: No DR, Mild, Moderate, Severe, and Proliferative DR. The network outputs a probability distribution across all 5 classes."
    },
    {
      icon: <Activity className="w-6 h-6 text-rose-400" />,
      title: "5. Explainability (Grad-CAM)",
      description: "Deep learning models are often 'black boxes'. To provide clinical trust, we implemented Gradient-weighted Class Activation Mapping (Grad-CAM). It traces the gradients of the predicted class back to the final convolutional layer to produce a heatmap highlighting the exact retinal regions that triggered the prediction."
    }
  ];

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 max-w-4xl mx-auto space-y-12">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl mb-4">
          How It Works
        </h1>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          Understanding the complete machine learning pipeline powering the RetinaAI platform.
        </p>
      </div>

      {/* Pipeline Visualization */}
      <div className="glass-card p-8 mb-12 overflow-hidden relative">
        <div className="absolute top-0 right-0 w-64 h-64 bg-sky-500/5 rounded-full blur-3xl" />
        <h2 className="text-xl font-bold text-white mb-8 flex items-center">
          <Network className="w-5 h-5 mr-3 text-sky-400" />
          System Architecture
        </h2>
        
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 relative z-10">
          <div className="bg-gray-900 border border-gray-800 p-4 rounded-xl text-center w-full md:w-1/4 z-10">
            <div className="w-12 h-12 bg-sky-500/20 text-sky-400 rounded-full flex items-center justify-center mx-auto mb-3">1</div>
            <p className="font-semibold text-gray-200 text-sm">Upload Scan</p>
          </div>
          
          <div className="hidden md:block h-px bg-gray-800 flex-1 relative top-[-10px]">
            <div className="absolute right-0 top-[-4px] w-2 h-2 border-t-2 border-r-2 border-gray-600 rotate-45" />
          </div>
          
          <div className="bg-gray-900 border border-gray-800 p-4 rounded-xl text-center w-full md:w-1/4 z-10">
            <div className="w-12 h-12 bg-teal-500/20 text-teal-400 rounded-full flex items-center justify-center mx-auto mb-3">2</div>
            <p className="font-semibold text-gray-200 text-sm">Preprocess</p>
          </div>
          
          <div className="hidden md:block h-px bg-gray-800 flex-1 relative top-[-10px]">
            <div className="absolute right-0 top-[-4px] w-2 h-2 border-t-2 border-r-2 border-gray-600 rotate-45" />
          </div>
          
          <div className="bg-gray-900 border border-sky-500/30 p-4 rounded-xl text-center w-full md:w-1/4 z-10 relative overflow-hidden group">
            <div className="absolute inset-0 bg-sky-500/5 group-hover:bg-sky-500/10 transition-colors" />
            <div className="w-12 h-12 bg-indigo-500/20 text-indigo-400 rounded-full flex items-center justify-center mx-auto mb-3">3</div>
            <p className="font-semibold text-white text-sm">CNN/ViT Inference</p>
          </div>
          
          <div className="hidden md:block h-px bg-gray-800 flex-1 relative top-[-10px]">
            <div className="absolute right-0 top-[-4px] w-2 h-2 border-t-2 border-r-2 border-gray-600 rotate-45" />
          </div>
          
          <div className="bg-gray-900 border border-gray-800 p-4 rounded-xl text-center w-full md:w-1/4 z-10">
            <div className="w-12 h-12 bg-rose-500/20 text-rose-400 rounded-full flex items-center justify-center mx-auto mb-3">4</div>
            <p className="font-semibold text-gray-200 text-sm">Grad-CAM + Result</p>
          </div>
        </div>
      </div>

      {/* Step by Step Details */}
      <div className="space-y-6 relative before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-gray-800 before:to-transparent">
        {steps.map((step, idx) => (
          <div 
            key={idx}
            className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active"
          >
            {/* Icon marker */}
            <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-gray-950 bg-gray-900 text-slate-500 group-[.is-active]:bg-gray-800 shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 relative z-10">
              {step.icon}
            </div>
            
            {/* Card */}
            <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] glass-card p-6">
              <h3 className="text-xl font-bold text-white mb-3">{step.title}</h3>
              <p className="text-gray-400 leading-relaxed text-sm">
                {step.description}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
