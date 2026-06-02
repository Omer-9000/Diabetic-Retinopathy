import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- MODEL SETUP ---
def get_model():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )
    )
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r'c:\Users\AK\Documents\dr\dr_finetuned.pth'

# Load the model
model = get_model()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.to(device).eval()

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

THRESHOLD = 5.16

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        # Load and preprocess image
        img = Image.open(file.stream).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            score = model(img_tensor).item()
            
        # Decision Logic based on test_model.py
        # Score > THRESHOLD -> Healthy / Mild (Class 0-2)
        # Score <= THRESHOLD -> Severe Diabetic Retinopathy (Class 3-6)
        
        is_diabetic = score <= THRESHOLD
        
        return jsonify({
            'score': round(score, 3),
            'threshold': THRESHOLD,
            'is_diabetic': is_diabetic,
            'message': 'High Risk of Diabetic Retinopathy' if is_diabetic else 'Healthy / Low Risk',
            'severity_label': 'Diabetic' if is_diabetic else 'Healthy'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure templates folder exists
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
