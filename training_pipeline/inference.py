import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from models import get_model

# Try importing Grad-CAM, handle if not installed
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False
    print("Warning: pytorch-grad-cam not installed. Grad-CAM visualizations will be disabled.")

CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
MODELS_DIR = "models"
RESULTS_DIR = "results"

def load_best_model(device):
    config_path = os.path.join(RESULTS_DIR, "best_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError("best_config.json not found. Run train.py first.")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    best_model_name = config['best_model']
    print(f"Loading best model: {best_model_name}")
    
    model = get_model(best_model_name, num_classes=5, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model.pth"), map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, best_model_name

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image).unsqueeze(0)
    return tensor, image

def predict(image_path, generate_cam=False, cam_save_path="gradcam_output.jpg"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name = load_best_model(device)
    
    tensor, original_image = preprocess_image(image_path)
    tensor = tensor.to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        pred_class_idx = np.argmax(probs)
    
    predicted_class = CLASS_NAMES[pred_class_idx]
    confidence = probs[pred_class_idx]
    
    result = {
        "Predicted Class": predicted_class,
        "Confidence Score": float(confidence),
        "Probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(5)}
    }
    
    if generate_cam and HAS_GRAD_CAM:
        # Determine the target layer for Grad-CAM based on model architecture
        target_layers = None
        if 'resnet' in model_name:
            target_layers = [model.layer4[-1]]
        elif 'densenet' in model_name:
            target_layers = [model.features[-1]]
        elif 'efficientnet' in model_name or 'mobilenet' in model_name:
            target_layers = [model.features[-1]]
        # ViT requires specific attention rollout implementations, Grad-CAM is tricky there out-of-the-box
        
        if target_layers is not None:
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
            targets = [ClassifierOutputTarget(pred_class_idx)]
            
            # You have to ensure tensor requires grad for Grad-CAM
            tensor.requires_grad_(True)
            
            grayscale_cam = cam(input_tensor=tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Prepare original image for overlay
            img_resized = np.array(original_image.resize((224, 224))) / 255.0
            visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
            
            cv2.imwrite(cam_save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            result["Grad-CAM Path"] = cam_save_path
        else:
            print(f"Grad-CAM is not natively supported for {model_name} in this script.")
            
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on an image")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--cam", action="store_true", help="Generate Grad-CAM explanation")
    
    args = parser.parse_args()
    
    try:
        res = predict(args.image_path, generate_cam=args.cam)
        print("\n--- Inference Results ---")
        print(json.dumps(res, indent=4))
    except Exception as e:
        print(f"Error during inference: {e}")
