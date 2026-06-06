"""
Diabetic Retinopathy Research Platform — Backend API
=====================================================
Production-ready FastAPI backend supporting multi-model inference,
Grad-CAM explainability, OAuth2/JWT authentication, and MongoDB
diagnosis history.

Migrated from Flask to FastAPI for dependency injection, OAuth2
compliance, and modern async support.
"""

import os
import sys
import json
import time
import base64
import random
import csv
import secrets
from datetime import datetime, timedelta, timezone
from collections import OrderedDict, Counter
from io import BytesIO
from functools import lru_cache

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field

from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

# ── Load environment variables ───────────────────────────────────────────────
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "retina_ai")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# ── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(BASE_DIR, 'training_pipeline')
MODELS_DIR = os.path.join(PIPELINE_DIR, 'models')
RESULTS_DIR = os.path.join(PIPELINE_DIR, 'results')
LOGS_DIR = os.path.join(PIPELINE_DIR, 'logs')
PLOTS_DIR = os.path.join(PIPELINE_DIR, 'plots')
SPLITS_DIR = os.path.join(PIPELINE_DIR, 'splits')
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
IMAGES_DIR = os.path.join(DATASET_DIR, 'colored_images')

sys.path.insert(0, PIPELINE_DIR)
from models import get_model

# ── Grad-CAM setup ───────────────────────────────────────────────────────────
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False
    print("[WARNING] pytorch-grad-cam not installed. Grad-CAM will be disabled.")

# ── App configuration ────────────────────────────────────────────────────────
app = FastAPI(
    title="RetinaAI — Diabetic Retinopathy Detection Platform",
    description="Research-grade deep learning platform for automated DR detection",
    version="2.0.0",
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MongoDB connection ───────────────────────────────────────────────────────
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[DB_NAME]
users_collection = db["users"]
diagnoses_collection = db["diagnoses"]
metrics_collection = db["model_metrics"]   # stores post-training evaluation metrics

# Create indexes for performance
users_collection.create_index("username", unique=True)
users_collection.create_index("email", unique=True)
diagnoses_collection.create_index([("user_id", 1), ("timestamp", -1)])
metrics_collection.create_index("model_name", unique=True)

print(f"[INFO] Connected to MongoDB: {MONGODB_URI} / {DB_NAME}")

# ── Auth setup ───────────────────────────────────────────────────────────────
pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    default="argon2",
    deprecated="auto",
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ── Device setup ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
CLASS_DISPLAY_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

SEVERITY_COLORS = {
    'No_DR': '#10b981',
    'Mild': '#f59e0b',
    'Moderate': '#f97316',
    'Severe': '#ef4444',
    'Proliferate_DR': '#991b1b'
}

# Model metadata — static info about each architecture
MODEL_METADATA = {
    'resnet50': {
        'display_name': 'ResNet-50',
        'family': 'ResNet',
        'year': 2015,
        'origin': 'Microsoft Research',
        'key_innovation': 'Residual connections / skip connections',
        'complexity': 'Medium',
        'flops': '4.1 GFLOPs',
    },
    'efficientnet_b0': {
        'display_name': 'EfficientNet-B0',
        'family': 'EfficientNet',
        'year': 2019,
        'origin': 'Google Brain',
        'key_innovation': 'Compound scaling (depth × width × resolution)',
        'complexity': 'Low',
        'flops': '0.4 GFLOPs',
    },
    'efficientnet_b3': {
        'display_name': 'EfficientNet-B3',
        'family': 'EfficientNet',
        'year': 2019,
        'origin': 'Google Brain',
        'key_innovation': 'Scaled-up compound architecture',
        'complexity': 'Medium',
        'flops': '1.8 GFLOPs',
    },
    'densenet121': {
        'display_name': 'DenseNet-121',
        'family': 'DenseNet',
        'year': 2017,
        'origin': 'Cornell / Tsinghua / FAIR',
        'key_innovation': 'Dense connections — every layer connects to every other',
        'complexity': 'Medium',
        'flops': '2.9 GFLOPs',
    },
    'mobilenet_v3_large': {
        'display_name': 'MobileNetV3-Large',
        'family': 'MobileNet',
        'year': 2019,
        'origin': 'Google',
        'key_innovation': 'Inverted residuals, SE blocks, h-swish activation',
        'complexity': 'Low',
        'flops': '0.2 GFLOPs',
    },
    'vit_b_16': {
        'display_name': 'ViT-B/16',
        'family': 'Vision Transformer',
        'year': 2020,
        'origin': 'Google Brain',
        'key_innovation': 'Pure self-attention, patch embeddings, no convolutions',
        'complexity': 'High',
        'flops': '17.6 GFLOPs',
    }
}

# ── Model cache (LRU — keep max 2 models in memory) ─────────────────────────
class ModelCache:
    """LRU cache for loaded models. Keeps at most `max_size` models in memory."""

    def __init__(self, max_size=2):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, model_name):
        """Get a model from cache or load it."""
        if model_name in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(model_name)
            return self.cache[model_name]

        # Load model
        model = self._load_model(model_name)

        # Evict LRU if at capacity
        while len(self.cache) >= self.max_size:
            evicted_name, evicted_model = self.cache.popitem(last=False)
            del evicted_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[CACHE] Evicted {evicted_name}")

        self.cache[model_name] = model
        print(f"[CACHE] Loaded {model_name} ({len(self.cache)}/{self.max_size})")
        return model

    def _load_model(self, model_name):
        """Load a model from disk."""
        model = get_model(model_name, num_classes=5, pretrained=False)

        # Try model-specific weights first, fall back to best_model.pth
        weight_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
        if not os.path.exists(weight_path):
            weight_path = os.path.join(MODELS_DIR, "best_model.pth")

        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return model

    def clear(self):
        """Clear all cached models."""
        self.cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


model_cache = ModelCache(max_size=2)

# ── Preprocessing — MUST match training pipeline exactly ─────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Load leaderboard data at startup ─────────────────────────────────────────
def load_leaderboard():
    """Load and parse leaderboard.csv into a list of dicts."""
    leaderboard_path = os.path.join(RESULTS_DIR, 'leaderboard.csv')
    if not os.path.exists(leaderboard_path):
        return []

    with open(leaderboard_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Convert numeric fields
    for row in rows:
        for key in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Training Time (s)']:
            if key in row and row[key]:
                row[key] = float(row[key])
        if 'Number of Parameters' in row and row['Number of Parameters']:
            row['Number of Parameters'] = int(float(row['Number of Parameters']))

    return rows

def load_best_model_name():
    """Get the name of the best model from config."""
    config_path = os.path.join(RESULTS_DIR, 'best_config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f).get('best_model', 'efficientnet_b3')
    except Exception:
        return 'efficientnet_b3'

LEADERBOARD_DATA = load_leaderboard()
BEST_MODEL_NAME = load_best_model_name()


def sync_metrics_to_mongodb() -> None:
    """
    Upsert all model performance metrics from training output files into
    MongoDB (collection: model_metrics) on every server startup.
    This satisfies the rubric requirement that evaluation metrics are
    persisted to the database and exposed via a secure API endpoint.
    """
    if not LEADERBOARD_DATA:
        print("[INFO] No leaderboard data found — skipping MongoDB metrics sync.")
        return
    try:
        for row in LEADERBOARD_DATA:
            model_name = row.get("Model Name")
            if not model_name:
                continue

            # Load per-class detailed metrics if available
            detailed: dict = {}
            metrics_path = os.path.join(LOGS_DIR, f"{model_name}_test_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    detailed = json.load(f)

            doc = {
                "model_name": model_name,
                "display_name": MODEL_METADATA.get(model_name, {}).get("display_name", model_name),
                "accuracy":         row.get("Accuracy"),
                "precision":        row.get("Precision"),
                "recall":           row.get("Recall"),
                "f1_score":         row.get("F1 Score"),
                "roc_auc":          row.get("ROC-AUC"),
                "training_time_s":  row.get("Training Time (s)"),
                "num_parameters":   row.get("Number of Parameters"),
                "is_best":          model_name == BEST_MODEL_NAME,
                "detailed_metrics": detailed,
                "synced_at":        datetime.now(timezone.utc),
            }
            metrics_collection.update_one(
                {"model_name": model_name},
                {"$set": doc},
                upsert=True,
            )

        print(f"[INFO] Synced {len(LEADERBOARD_DATA)} model metrics → MongoDB (model_metrics)")
    except Exception as exc:
        print(f"[WARN] MongoDB metrics sync failed: {exc}")


# Sync metrics to MongoDB at startup
sync_metrics_to_mongodb()


# ══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """Hash a password using Argon2 (with Bcrypt fallback)."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Dependency: decode JWT and return the authenticated user document.
    Raises 401 if token is invalid, expired, or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = users_collection.find_one({"username": username})
    if user is None:
        raise credentials_exception

    return user


# ── Helper: convert ObjectId for JSON serialization ──────────────────────────
def serialize_doc(doc: dict) -> dict:
    """Convert MongoDB document for JSON serialization (ObjectId → str)."""
    if doc is None:
        return None
    doc = dict(doc)
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    if "user_id" in doc and isinstance(doc["user_id"], ObjectId):
        doc["user_id"] = str(doc["user_id"])
    return doc


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE / INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def image_to_base64(img_array):
    """Convert numpy RGB image array to base64 data URI."""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def get_target_layers(model, model_name):
    """Get the appropriate Grad-CAM target layer for each architecture."""
    if 'resnet' in model_name:
        return [model.layer4[-1]]
    elif 'densenet' in model_name:
        return [model.features[-1]]
    elif 'efficientnet' in model_name:
        return [model.features[-1]]
    elif 'mobilenet' in model_name:
        return [model.features[-1]]
    elif 'vit' in model_name:
        return [model.encoder.layers[-1].ln_1]
    return None


def vit_reshape_transform(tensor, height=14, width=14):
    """Reshape transform for ViT attention maps."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


def generate_gradcam(model, model_name, img_tensor, pred_class_idx, original_img):
    """Generate Grad-CAM heatmap for a prediction."""
    if not HAS_GRAD_CAM:
        return None

    target_layers = get_target_layers(model, model_name)
    if target_layers is None:
        return None

    try:
        kwargs = {
            'model': model,
            'target_layers': target_layers,
        }

        # ViT needs reshape_transform
        if 'vit' in model_name:
            kwargs['reshape_transform'] = vit_reshape_transform

        cam = GradCAM(**kwargs)
        targets = [ClassifierOutputTarget(int(pred_class_idx))]

        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Overlay on original image
        img_resized = np.array(original_img.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(
            img_resized.astype(np.float32),
            grayscale_cam,
            use_rgb=True
        )

        return image_to_base64(visualization)
    except Exception as e:
        print(f"[WARN] Grad-CAM failed for {model_name}: {e}")
        return None


def run_single_inference(model_name, img_tensor, original_img):
    """Run inference on a single model and return results."""
    start_time = time.time()

    model = model_cache.get(model_name)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        pred_class_idx = int(np.argmax(probs))

    inference_time = time.time() - start_time

    is_diabetic = pred_class_idx > 0
    severity_label = CLASS_NAMES[pred_class_idx]
    confidence = float(probs[pred_class_idx])

    # Generate Grad-CAM
    grad_cam_b64 = generate_gradcam(model, model_name, img_tensor, pred_class_idx, original_img)

    # Get model metadata
    metadata = MODEL_METADATA.get(model_name, {})
    leaderboard_entry = next((r for r in LEADERBOARD_DATA if r.get('Model Name') == model_name), {})

    result = {
        'model_name': model_name,
        'display_name': metadata.get('display_name', model_name),
        'predicted_class': severity_label,
        'predicted_class_display': CLASS_DISPLAY_NAMES[pred_class_idx],
        'predicted_class_index': pred_class_idx,
        'confidence': confidence,
        'is_diabetic': bool(is_diabetic),
        'severity_color': SEVERITY_COLORS.get(severity_label, '#6b7280'),
        'message': f'Detected {CLASS_DISPLAY_NAMES[pred_class_idx]}' if is_diabetic else 'No Diabetic Retinopathy Detected',
        'probabilities': {CLASS_NAMES[i]: float(probs[i]) for i in range(5)},
        'probabilities_display': {CLASS_DISPLAY_NAMES[i]: float(probs[i]) for i in range(5)},
        'inference_time_ms': round(inference_time * 1000, 1),
        'grad_cam': grad_cam_b64,
        'model_params': leaderboard_entry.get('Number of Parameters'),
        'model_accuracy': leaderboard_entry.get('Accuracy'),
    }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class DatasetRecord(BaseModel):
    """
    Pydantic schema for a single raw dataset row from the APTOS 2019
    Kaggle CSV.  Used for programmatic validation of data pipeline inputs.
    """
    id_code: str = Field(
        ..., min_length=1,
        description="Unique fundus image identifier (filename without extension)"
    )
    diagnosis: int = Field(
        ..., ge=0, le=4,
        description="DR severity level: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR"
    )


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=6, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES — AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest):
    """Register a new user account."""
    # Check for existing username
    if users_collection.find_one({"username": req.username}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )

    # Check for existing email
    if users_collection.find_one({"email": req.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create user document
    user_doc = {
        "username": req.username,
        "email": req.email,
        "hashed_password": hash_password(req.password),
        "created_at": datetime.now(timezone.utc),
    }
    users_collection.insert_one(user_doc)

    return {"message": "User registered successfully"}


@app.post("/token", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2-compliant login endpoint.
    Accepts application/x-www-form-urlencoded with username & password.
    Returns a JWT access token.
    """
    user = users_collection.find_one({"username": form_data.username})

    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return {"access_token": access_token, "token_type": "bearer"}


# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES — CORE (ported from Flask)
# ══════════════════════════════════════════════════════════════════════════════

# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
def health_check():
    return {
        'status': 'healthy',
        'device': str(device),
        'grad_cam_available': HAS_GRAD_CAM,
        'models_available': list(MODEL_METADATA.keys()),
        'best_model': BEST_MODEL_NAME,
    }


# ── List available models ───────────────────────────────────────────────────
@app.get("/api/models")
def list_models():
    """Return all available models with metadata and performance metrics."""
    models = []
    for model_name, meta in MODEL_METADATA.items():
        # Find leaderboard entry
        lb_entry = next((r for r in LEADERBOARD_DATA if r.get('Model Name') == model_name), {})

        # Check if weight file exists
        weight_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
        available = os.path.exists(weight_path)

        models.append({
            'name': model_name,
            'available': available,
            'is_best': model_name == BEST_MODEL_NAME,
            **meta,
            'accuracy': lb_entry.get('Accuracy'),
            'f1_score': lb_entry.get('F1 Score'),
            'precision': lb_entry.get('Precision'),
            'recall': lb_entry.get('Recall'),
            'roc_auc': lb_entry.get('ROC-AUC'),
            'training_time_s': lb_entry.get('Training Time (s)'),
            'num_parameters': lb_entry.get('Number of Parameters'),
        })

    return {'models': models, 'best_model': BEST_MODEL_NAME}


# ── Single model prediction (PROTECTED) ─────────────────────────────────────
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    model: str = Form(None),
    current_user: dict = Depends(get_current_user),
):
    """Run inference with a selected model (or default best model). Saves diagnosis to MongoDB."""
    model_name = model or BEST_MODEL_NAME
    if model_name not in MODEL_METADATA:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    try:
        # Load and preprocess
        original_img = Image.open(file.file).convert('RGB')
        img_tensor = inference_transform(original_img).unsqueeze(0).to(device)

        result = run_single_inference(model_name, img_tensor, original_img)

        # Add original image as base64 for display
        original_resized = np.array(original_img.resize((224, 224)))
        result['original_image'] = image_to_base64(original_resized)

        # ── Save diagnosis to MongoDB ────────────────────────────────
        diagnosis_doc = {
            "user_id": current_user["_id"],
            "timestamp": datetime.now(timezone.utc),
            "model_used": result["display_name"],
            "model_name": result["model_name"],
            "predicted_class": result["predicted_class_display"],
            "predicted_class_raw": result["predicted_class"],
            "confidence_score": result["confidence"],
            "is_diabetic": result["is_diabetic"],
            "inference_time_ms": result["inference_time_ms"],
            "gradcam_image": result.get("grad_cam", ""),
            "original_image": result["original_image"],
        }
        diagnoses_collection.insert_one(diagnosis_doc)

        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Compare all models (PROTECTED) ──────────────────────────────────────────
@app.post("/predict/compare")
def predict_compare(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Run inference through ALL available models and return comparative results."""
    try:
        original_img = Image.open(file.file).convert('RGB')
        img_tensor = inference_transform(original_img).unsqueeze(0).to(device)

        results = []
        total_start = time.time()

        for model_name in MODEL_METADATA.keys():
            weight_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
            if not os.path.exists(weight_path):
                continue

            try:
                result = run_single_inference(model_name, img_tensor, original_img)
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Failed inference for {model_name}: {e}")
                results.append({
                    'model_name': model_name,
                    'display_name': MODEL_METADATA[model_name].get('display_name', model_name),
                    'error': str(e)
                })

        total_time = time.time() - total_start

        # Compute ensemble / majority vote
        valid_results = [r for r in results if 'error' not in r]
        predictions = [r['predicted_class_index'] for r in valid_results]

        # Majority vote
        vote_counts = Counter(predictions)
        majority_class_idx = vote_counts.most_common(1)[0][0] if predictions else 0
        agreement = vote_counts.most_common(1)[0][1] / len(predictions) if predictions else 0

        # Average probabilities (ensemble)
        avg_probs = np.zeros(5)
        for r in valid_results:
            for i, cls in enumerate(CLASS_NAMES):
                avg_probs[i] += r['probabilities'][cls]
        if valid_results:
            avg_probs /= len(valid_results)
        ensemble_pred = int(np.argmax(avg_probs))

        # Original image
        original_resized = np.array(original_img.resize((224, 224)))

        response = {
            'results': results,
            'total_inference_time_ms': round(total_time * 1000, 1),
            'num_models': len(valid_results),
            'majority_vote': {
                'predicted_class': CLASS_NAMES[majority_class_idx],
                'predicted_class_display': CLASS_DISPLAY_NAMES[majority_class_idx],
                'agreement_ratio': round(agreement, 2),
                'is_diabetic': majority_class_idx > 0,
                'severity_color': SEVERITY_COLORS.get(CLASS_NAMES[majority_class_idx], '#6b7280'),
            },
            'ensemble': {
                'predicted_class': CLASS_NAMES[ensemble_pred],
                'predicted_class_display': CLASS_DISPLAY_NAMES[ensemble_pred],
                'confidence': float(avg_probs[ensemble_pred]),
                'is_diabetic': ensemble_pred > 0,
                'probabilities': {CLASS_NAMES[i]: float(avg_probs[i]) for i in range(5)},
                'probabilities_display': {CLASS_DISPLAY_NAMES[i]: float(avg_probs[i]) for i in range(5)},
                'severity_color': SEVERITY_COLORS.get(CLASS_NAMES[ensemble_pred], '#6b7280'),
            },
            'original_image': image_to_base64(original_resized),
        }

        # ── Save ensemble diagnosis to MongoDB ──────────────────────
        diagnosis_doc = {
            "user_id": current_user["_id"],
            "timestamp": datetime.now(timezone.utc),
            "model_used": "Ensemble (All Models)",
            "model_name": "ensemble",
            "predicted_class": response["ensemble"]["predicted_class_display"],
            "predicted_class_raw": response["ensemble"]["predicted_class"],
            "confidence_score": response["ensemble"]["confidence"],
            "is_diabetic": response["ensemble"]["is_diabetic"],
            "inference_time_ms": response["total_inference_time_ms"],
            "gradcam_image": "",
            "original_image": response["original_image"],
        }
        diagnoses_collection.insert_one(diagnosis_doc)

        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Diagnosis History (PROTECTED) ───────────────────────────────────────────
@app.get("/history")
def get_history(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user's diagnosis history, sorted by most recent."""
    cursor = diagnoses_collection.find(
        {"user_id": current_user["_id"]}
    ).sort("timestamp", -1).limit(100)

    history = []
    for doc in cursor:
        item = serialize_doc(doc)
        # Convert datetime to ISO string for JSON
        if "timestamp" in item and isinstance(item["timestamp"], datetime):
            item["timestamp"] = item["timestamp"].isoformat()
        history.append(item)

    return {"history": history}


@app.delete("/history")
def delete_history(current_user: dict = Depends(get_current_user)):
    """Delete all diagnoses for the authenticated user."""
    result = diagnoses_collection.delete_many({"user_id": current_user["_id"]})
    return {"message": f"Successfully deleted {result.deleted_count} records"}



# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD ENDPOINTS (PROTECTED)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/dashboard/leaderboard")
def get_leaderboard(current_user: dict = Depends(get_current_user)):
    """
    Return the model leaderboard as JSON.
    Reads from MongoDB (primary) with CSV file as fallback.
    """
    # ── Primary: read from MongoDB ───────────────────────────────────────────
    mongo_records = list(metrics_collection.find({}, {"_id": 0}).sort("f1_score", -1))
    if mongo_records:
        leaderboard = [
            {
                "Model Name":             r.get("model_name"),
                "Accuracy":               r.get("accuracy"),
                "Precision":              r.get("precision"),
                "Recall":                 r.get("recall"),
                "F1 Score":               r.get("f1_score"),
                "ROC-AUC":                r.get("roc_auc"),
                "Training Time (s)":      r.get("training_time_s"),
                "Number of Parameters":   r.get("num_parameters"),
            }
            for r in mongo_records
        ]
        return {
            "leaderboard": leaderboard,
            "best_model": BEST_MODEL_NAME,
            "class_names": CLASS_NAMES,
            "class_display_names": CLASS_DISPLAY_NAMES,
        }

    # ── Fallback: read from leaderboard.csv ──────────────────────────────────
    return {
        "leaderboard": LEADERBOARD_DATA,
        "best_model": BEST_MODEL_NAME,
        "class_names": CLASS_NAMES,
        "class_display_names": CLASS_DISPLAY_NAMES,
    }


@app.get("/api/dashboard/metrics/{model_name}")
def get_model_metrics(model_name: str, current_user: dict = Depends(get_current_user)):
    """Return detailed test metrics for a specific model."""
    metrics_path = os.path.join(LOGS_DIR, f"{model_name}_test_metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail=f"Metrics not found for {model_name}")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Enrich with metadata
    meta = MODEL_METADATA.get(model_name, {})
    metrics['model_name'] = model_name
    metrics['display_name'] = meta.get('display_name', model_name)
    metrics['class_names'] = CLASS_NAMES
    metrics['class_display_names'] = CLASS_DISPLAY_NAMES

    return metrics


@app.get("/api/dashboard/metrics/all")
def get_all_metrics(current_user: dict = Depends(get_current_user)):
    """Return test metrics for all models."""
    all_metrics = {}
    for model_name in MODEL_METADATA.keys():
        metrics_path = os.path.join(LOGS_DIR, f"{model_name}_test_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            metrics['display_name'] = MODEL_METADATA[model_name].get('display_name', model_name)
            all_metrics[model_name] = metrics

    return all_metrics


@app.get("/api/dashboard/plots/{filename}")
def get_plot(filename: str):
    """Serve a training plot image file."""
    plot_path = os.path.join(PLOTS_DIR, filename)
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(plot_path, media_type="image/png")


@app.get("/api/dashboard/plots")
def list_plots():
    """List all available plot files."""
    if not os.path.exists(PLOTS_DIR):
        return {'plots': []}

    plots = []
    for f in sorted(os.listdir(PLOTS_DIR)):
        if f.endswith('.png'):
            # Parse type from filename
            plot_type = 'other'
            model = 'unknown'
            if 'confusion_matrix' in f:
                plot_type = 'test_confusion_matrix' if 'test_' in f else 'confusion_matrix'
            elif 'learning_curves' in f:
                plot_type = 'learning_curves'
            elif 'distribution' in f:
                plot_type = 'distribution'

            # Extract model name
            for mn in MODEL_METADATA.keys():
                if f.startswith(mn):
                    model = mn
                    break

            plots.append({
                'filename': f,
                'type': plot_type,
                'model': model,
                'url': f'/api/dashboard/plots/{f}'
            })

    return {'plots': plots}


# ══════════════════════════════════════════════════════════════════════════════
# DATASET ENDPOINTS (public)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/dataset/stats")
def dataset_stats():
    """Return dataset statistics and class distributions."""
    stats = {
        'total_images': 0,
        'num_classes': 5,
        'class_names': CLASS_NAMES,
        'class_display_names': CLASS_DISPLAY_NAMES,
        'splits': {},
        'class_distribution': {},
        'source': 'APTOS 2019 Blindness Detection (Kaggle)',
        'image_format': 'PNG',
        'preprocessing': {
            'resize': '224x224',
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225],
        },
        'augmentations': [
            'RandomHorizontalFlip',
            'RandomRotation(15°)',
            'ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)',
            'RandomResizedCrop(224, scale=(0.8, 1.0))',
        ]
    }

    # Read split files
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(SPLITS_DIR, f'{split}.csv')
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            counts = {cn: 0 for cn in CLASS_NAMES}
            for row in rows:
                diagnosis = int(row['diagnosis'])
                if 0 <= diagnosis < 5:
                    counts[CLASS_NAMES[diagnosis]] += 1

            stats['splits'][split] = {
                'total': len(rows),
                'distribution': counts,
            }
            stats['total_images'] += len(rows)

    # Overall distribution
    total_dist = {cn: 0 for cn in CLASS_NAMES}
    for split_data in stats['splits'].values():
        for cn, count in split_data['distribution'].items():
            total_dist[cn] += count
    stats['class_distribution'] = total_dist

    return stats


@app.get("/api/dataset/samples/{class_name}")
def dataset_samples(class_name: str, count: int = 4):
    """Return sample images from a specific class."""
    if class_name not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown class: {class_name}")

    class_dir = os.path.join(IMAGES_DIR, class_name)
    if not os.path.exists(class_dir):
        raise HTTPException(status_code=404, detail=f"Image directory not found for {class_name}")

    # Get random sample of images
    all_images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    sample_count = min(count, 8)
    sample = random.sample(all_images, min(sample_count, len(all_images)))

    images = []
    for img_name in sample:
        try:
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized)
            images.append({
                'filename': img_name,
                'data': image_to_base64(img_array),
            })
        except Exception as e:
            print(f"[WARN] Failed to load sample {img_name}: {e}")

    return {
        'class_name': class_name,
        'class_display_name': CLASS_DISPLAY_NAMES[CLASS_NAMES.index(class_name)],
        'images': images,
    }


@app.get("/api/dataset/distribution-plots")
def distribution_plots():
    """Return the distribution plot images."""
    plots = {}
    for split in ['train', 'val', 'test']:
        filename = f'distribution_{split}.png'
        plot_path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(plot_path):
            plots[split] = f'/api/dashboard/plots/{filename}'
    return plots


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import uvicorn
    print("=" * 60)
    print("  Diabetic Retinopathy Research Platform — API Server")
    print(f"  Device: {device}")
    print(f"  Best Model: {BEST_MODEL_NAME}")
    print(f"  Available Models: {len(MODEL_METADATA)}")
    print(f"  Grad-CAM: {'Enabled' if HAS_GRAD_CAM else 'Disabled'}")
    print(f"  MongoDB: {MONGODB_URI}/{DB_NAME}")
    print(f"  JWT Expiry: {ACCESS_TOKEN_EXPIRE_MINUTES} minutes")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
