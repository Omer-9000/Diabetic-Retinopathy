# RetinaAI — Setup Guide

> **Project:** RetinaAI — Deep Learning Platform for Diabetic Retinopathy Detection  
> **Companion Document:** [REQUIREMENTS.md](./REQUIREMENTS.md)  
> **Target Audience:** New developers onboarding to the project

---

## Table of Contents

- [1. Install Git & Git LFS](#1-install-git--git-lfs)
- [2. Clone the Repository](#2-clone-the-repository)
- [3. Install Python](#3-install-python)
- [4. Create a Python Virtual Environment](#4-create-a-python-virtual-environment)
- [5. Install Backend Dependencies](#5-install-backend-dependencies)
- [6. Install Node.js & npm](#6-install-nodejs--npm)
- [7. Install Frontend Dependencies](#7-install-frontend-dependencies)
- [8. Set Up MongoDB](#8-set-up-mongodb)
- [9. Configure Environment Variables](#9-configure-environment-variables)
- [10. Download Model Weights (Git LFS)](#10-download-model-weights-git-lfs)
- [11. Download the Dataset (Training Only)](#11-download-the-dataset-training-only)
- [12. Run the Backend Server](#12-run-the-backend-server)
- [13. Run the Frontend Dev Server](#13-run-the-frontend-dev-server)
- [14. Run the Training Pipeline](#14-run-the-training-pipeline)
- [15. Run the Preprocessing Pipeline](#15-run-the-preprocessing-pipeline)
- [16. Run the Standalone Inference Script](#16-run-the-standalone-inference-script)
- [17. Running Tests](#17-running-tests)
- [18. Verification Steps](#18-verification-steps)
- [19. Troubleshooting Common Issues](#19-troubleshooting-common-issues)

---

## 1. Install Git & Git LFS

Git is required for version control. **Git LFS** (Large File Storage) is required because the trained model weights (`.pth` files, ~558 MB total) are tracked via LFS.

### Windows

1. Download and install Git from [https://git-scm.com/download/win](https://git-scm.com/download/win).
2. During installation, ensure "Git LFS" is checked (it's included by default in modern Git for Windows installers).
3. Open a terminal (PowerShell or Git Bash) and verify:

```powershell
git --version
# Expected: git version 2.x.x

git lfs --version
# Expected: git-lfs/3.x.x
```

4. If Git LFS was not included, install it separately:

```powershell
git lfs install
```

### macOS

```bash
brew install git
brew install git-lfs
git lfs install
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install git git-lfs
git lfs install
```

---

## 2. Clone the Repository

```bash
# Clone the repository (Git LFS files will be downloaded as pointers initially)
git clone https://github.com/yourusername/diabetic-retinopathy.git
cd diabetic-retinopathy

# Download the actual model weight binaries
git lfs pull
```

> ⚠️ **Important:** If you skip `git lfs pull`, the `.pth` files in `training_pipeline/models/` will be small text pointer files instead of actual model weights, and the application will fail to load models.

### Verify LFS Files

```bash
git lfs ls-files
```

You should see 7 `.pth` files listed:

```
* training_pipeline/models/best_model.pth
* training_pipeline/models/densenet121_best.pth
* training_pipeline/models/efficientnet_b0_best.pth
* training_pipeline/models/efficientnet_b3_best.pth
* training_pipeline/models/mobilenet_v3_large_best.pth
* training_pipeline/models/resnet50_best.pth
* training_pipeline/models/vit_b_16_best.pth
```

---

## 3. Install Python

Python 3.10 or higher is required for the backend API and training pipeline.

### Windows

1. Download Python from [https://www.python.org/downloads/](https://www.python.org/downloads/) (version 3.10, 3.11, or 3.12 recommended).
2. **During installation, check "Add Python to PATH"**.
3. Verify:

```powershell
python --version
# Expected: Python 3.10.x or higher

pip --version
# Expected: pip 23.x or higher
```

### macOS

```bash
brew install python@3.12
python3 --version
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
python3 --version
```

---

## 4. Create a Python Virtual Environment

A virtual environment isolates the project's Python dependencies from your system Python.

```bash
# Navigate to the project root directory
cd diabetic-retinopathy

# Create a virtual environment
python -m venv venv
```

### Activate the Virtual Environment

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

> 💡 If you get an execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal prompt.

---

## 5. Install Backend Dependencies

With the virtual environment activated:

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Install all Python dependencies
pip install -r requirements.txt
```

### PyTorch with GPU Support (Recommended)

The default `pip install torch` installs the CPU-only version. For **GPU-accelerated inference and training**, install PyTorch with CUDA support:

```bash
# For CUDA 11.8 (check your GPU driver version first)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> 💡 Check your CUDA version with `nvidia-smi` and match the PyTorch install command accordingly. Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for the latest installation matrix.

### Verify PyTorch Installation

```python
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected output (with GPU):

```
PyTorch 2.x.x
CUDA available: True
Device: NVIDIA GeForce RTX 4060
```

### Install Argon2 Backend (if passlib[argon2] fails)

On some systems, the `argon2-cffi` package requires a C compiler. If installation fails:

**Windows:**

```powershell
pip install argon2-cffi
```

If this fails, install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

**Linux:**

```bash
sudo apt install build-essential python3-dev
pip install argon2-cffi
```

---

## 6. Install Node.js & npm

Node.js 18+ is required for the Next.js frontend.

### Windows

1. Download and install Node.js LTS from [https://nodejs.org/](https://nodejs.org/).
2. The installer includes npm automatically.
3. Verify:

```powershell
node --version
# Expected: v18.x.x or higher (v20.x.x or v22.x.x recommended)

npm --version
# Expected: 9.x.x or higher
```

### macOS

```bash
brew install node
```

### Linux (Ubuntu/Debian)

```bash
# Using NodeSource repository for latest LTS
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

node --version
npm --version
```

---

## 7. Install Frontend Dependencies

```bash
# Navigate to the frontend directory
cd frontend

# Install all Node.js packages
npm install
```

This will install all packages defined in `package.json` and create a `node_modules/` directory and `package-lock.json` file.

```bash
# Return to the project root
cd ..
```

---

## 8. Set Up MongoDB

MongoDB is required for user authentication and diagnosis history logging.

### Option A: MongoDB Atlas (Cloud — Recommended for Quick Start)

1. Go to [https://www.mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas) and create a free account.
2. Create a free-tier cluster (M0 Sandbox).
3. Under **Database Access**, create a database user with a username and password.
4. Under **Network Access**, add your current IP address (or `0.0.0.0/0` for development).
5. Click **Connect** → **Connect your application** → Copy the connection string.
6. The connection string will look like:

```
mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority&appName=RetinaAI
```

### Option B: Local MongoDB Installation

**Windows:**

1. Download MongoDB Community Server from [https://www.mongodb.com/try/download/community](https://www.mongodb.com/try/download/community).
2. Run the installer with default settings (install as a Windows Service).
3. MongoDB will run on `mongodb://localhost:27017` by default.
4. Optionally install [MongoDB Compass](https://www.mongodb.com/products/compass) for a GUI.

**macOS:**

```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

**Linux (Ubuntu/Debian):**

```bash
# Import MongoDB public GPG key
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor

# Add MongoDB repository
echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Install and start
sudo apt update
sudo apt install -y mongodb-org
sudo systemctl start mongod
sudo systemctl enable mongod
```

### Verify MongoDB is Running

```bash
# For local installation
mongosh --eval "db.runCommand({ ping: 1 })"
# Expected: { ok: 1 }
```

---

## 9. Configure Environment Variables

Create a `.env` file in the **project root directory** (same level as `app.py`):

```bash
# Windows (PowerShell)
New-Item -Path ".env" -ItemType File

# macOS / Linux
touch .env
```

Add the following content to `.env`:

```env
# ═══════════════════════════════════════════════
# RetinaAI — Environment Configuration
# ═══════════════════════════════════════════════

# ── MongoDB Connection ──────────────────────────
# For MongoDB Atlas (cloud):
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority&appName=RetinaAI

# For local MongoDB:
# MONGODB_URI=mongodb://localhost:27017

# Database name
DB_NAME=retina_ai

# ── JWT Authentication ──────────────────────────
# Generate a secure key: python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY=YOUR_SECRET_KEY_HERE
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Generate a Secure JWT Secret Key

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy the output and paste it as the value for `JWT_SECRET_KEY`.

> ⚠️ **Important:** The `.env` file is listed in `.gitignore` and should **never** be committed to version control.

---

## 10. Download Model Weights (Git LFS)

If you followed Step 2 correctly, model weights should already be downloaded. Verify:

```bash
# Check file sizes — they should NOT be tiny (pointer files are ~130 bytes)
ls -lh training_pipeline/models/

# Windows PowerShell equivalent:
Get-ChildItem training_pipeline\models\*.pth | Select-Object Name, @{N='Size(MB)';E={[math]::Round($_.Length/1MB,1)}}
```

Expected sizes:

| File                          | Expected Size |
| :---------------------------- | :------------ |
| `resnet50_best.pth`           | ~90 MB        |
| `efficientnet_b3_best.pth`    | ~41 MB        |
| `vit_b_16_best.pth`          | ~327 MB       |
| `densenet121_best.pth`        | ~27 MB        |
| `efficientnet_b0_best.pth`    | ~16 MB        |
| `mobilenet_v3_large_best.pth` | ~16 MB        |
| `best_model.pth`             | ~41 MB        |

If the files are tiny (~130 bytes), run:

```bash
git lfs pull
```

---

## 11. Download the Dataset (Training Only)

> ⏭️ **Skip this step** if you only want to run inference (the pre-trained models and evaluator samples are included in the repository).

To train models from scratch:

1. Go to [https://www.kaggle.com/c/aptos2019-blindness-detection](https://www.kaggle.com/c/aptos2019-blindness-detection).
2. Download `train.csv` and the image archive.
3. Organize the images into class subfolders:

```
Dataset/
├── train.csv
└── colored_images/
    ├── No_DR/
    │   ├── 000c1434d8d7.png
    │   └── ...
    ├── Mild/
    ├── Moderate/
    ├── Severe/
    └── Proliferate_DR/
```

4. Update the paths in `training_pipeline/train.py`:

```python
# Change these to your actual paths:
CSV_PATH = r"path/to/Dataset/train.csv"
IMAGE_DIR = r"path/to/Dataset/colored_images"
```

---

## 12. Run the Backend Server

Ensure your virtual environment is activated and your `.env` file is configured.

```bash
# From the project root directory
python app.py
```

### Expected Startup Output

```
[INFO] Connected to MongoDB: mongodb://localhost:27017 / retina_ai
[INFO] Using device: cuda
============================================================
  Diabetic Retinopathy Research Platform — API Server
  Device: cuda
  Best Model: efficientnet_b3
  Available Models: 6
  Grad-CAM: Enabled
  MongoDB: mongodb://localhost:27017/retina_ai
  JWT Expiry: 30 minutes
============================================================
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Verify Backend

Open a browser and navigate to:

- **Health Check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)
- **API Documentation (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Documentation (ReDoc):** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 13. Run the Frontend Dev Server

Open a **new terminal window** (keep the backend running in the first one):

```bash
# Navigate to the frontend directory
cd frontend

# Start the Next.js development server
npm run dev
```

### Expected Output

```
   ▲ Next.js 16.2.7
   - Local:        http://localhost:3000
   - Environments: .env.local

 ✓ Starting...
 ✓ Ready in X.Xs
```

Open your browser and navigate to [http://localhost:3000](http://localhost:3000).

---

## 14. Run the Training Pipeline

> ⚠️ **Prerequisites:** APTOS 2019 dataset downloaded and organized (see [Step 11](#11-download-the-dataset-training-only)), GPU highly recommended.

```bash
# Ensure virtual environment is activated
# Navigate to the training pipeline directory
cd training_pipeline

# Start training all 6 models
python train.py
```

### What the Training Pipeline Does

1. **Reads and splits** the dataset CSV into 70% train / 15% validation / 15% test.
2. **Generates class distribution** plots in `training_pipeline/plots/`.
3. **Trains 6 architectures** sequentially: ResNet-50, EfficientNet-B0, EfficientNet-B3, DenseNet-121, MobileNetV3-Large, ViT-B/16.
4. **Uses Focal Loss** with class weights, AdamW optimizer, ReduceLROnPlateau scheduler, and early stopping (patience=7).
5. **Evaluates each model** on the test set and saves metrics to `training_pipeline/logs/`.
6. **Generates confusion matrices** and learning curves in `training_pipeline/plots/`.
7. **Creates a leaderboard** CSV in `training_pipeline/results/leaderboard.csv`.
8. **Copies the best model** weights to `training_pipeline/models/best_model.pth`.

### Training Configuration

| Parameter       | Default Value | Notes                              |
| :-------------- | :------------ | :--------------------------------- |
| Batch Size      | 32            | Reduce to 16 if GPU OOM occurs    |
| Epochs          | 30            | Early stopping may trigger earlier |
| Patience        | 7             | Epochs before early stopping       |
| Learning Rate   | 1e-4          | AdamW optimizer                    |
| Image Size      | 224 × 224     | Standard ImageNet preprocessing    |

### Estimated Training Time

| Setup                      | Approximate Time (all 6 models) |
| :------------------------- | :------------------------------ |
| NVIDIA RTX 3060 (6 GB)     | ~35–45 minutes                  |
| NVIDIA RTX 4070 (12 GB)    | ~20–30 minutes                  |
| CPU Only (not recommended) | 6–12 hours                      |

---

## 15. Run the Preprocessing Pipeline

The preprocessing script enhances fundus images using a 10-step pipeline (green channel extraction, CLAHE, denoising, sharpening, morphological operations, gamma correction, and edge detection).

```bash
python "Test data/saving ts.py" --input "path/to/input/images" --output "path/to/output"
```

Or edit the default paths directly in the script and run:

```bash
python "Test data/saving ts.py"
```

---

## 16. Run the Standalone Inference Script

For quick command-line inference without running the full web application:

```bash
cd training_pipeline

# Basic inference
python inference.py path/to/retinal_image.png

# With Grad-CAM visualization
python inference.py path/to/retinal_image.png --cam
```

---

## 17. Running Tests

> ⚠️ **Note:** The project does not currently include a formal test suite. Below are manual verification steps.

### Backend API Smoke Test

With the backend running, use `curl` or PowerShell to test endpoints:

```bash
# Health check
curl http://localhost:8000/api/health

# List available models
curl http://localhost:8000/api/models

# Register a test user
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@example.com","password":"testpass123"}'

# Login and get a token
curl -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=testpass123"
```

### Frontend Lint Check

```bash
cd frontend
npm run lint
```

### Frontend Production Build Test

```bash
cd frontend
npm run build
```

---

## 18. Verification Steps

Use this checklist to confirm the project is correctly configured:

### ✅ Backend Verification

| Step | Check                                                           | Expected Result                        |
| :--- | :-------------------------------------------------------------- | :------------------------------------- |
| 1    | `python app.py` starts without errors                           | Server running on port 8000            |
| 2    | Visit `http://localhost:8000/api/health`                        | JSON with `"status": "healthy"`        |
| 3    | `"device"` field in health check                                | `"cuda"` (GPU) or `"cpu"`             |
| 4    | `"grad_cam_available"` in health check                          | `true`                                 |
| 5    | Visit `http://localhost:8000/docs`                              | Swagger UI loads with all endpoints    |
| 6    | `http://localhost:8000/api/models` lists 6 models               | All models with `"available": true`    |
| 7    | Register a user via `/register`                                 | `201 Created`                          |
| 8    | Login via `/token`                                              | Returns `access_token`                 |

### ✅ Frontend Verification

| Step | Check                                                           | Expected Result                        |
| :--- | :-------------------------------------------------------------- | :------------------------------------- |
| 1    | `npm run dev` starts without errors                             | Server running on port 3000            |
| 2    | Visit `http://localhost:3000`                                   | Landing page loads with RetinaAI branding |
| 3    | Navigate to Login page                                          | Login form displayed                   |
| 4    | Register and login                                              | Redirected to diagnostic workspace     |
| 5    | Upload a sample image from `evaluator_samples/`                 | Prediction result displayed            |
| 6    | Grad-CAM heatmap visible                                        | Color overlay on retinal image         |

### ✅ End-to-End Verification

1. Start the backend: `python app.py`
2. Start the frontend: `cd frontend && npm run dev`
3. Open [http://localhost:3000](http://localhost:3000)
4. Register a new user account
5. Login with the new credentials
6. Navigate to the Diagnostics Workspace
7. Upload an image from the `evaluator_samples/` folder (e.g., `000c1434d8d7.png`)
8. Select a model (or use default EfficientNet-B3)
9. Click **Analyze Image**
10. Verify that:
    - A severity class is displayed (e.g., "No DR", "Mild", "Moderate")
    - A confidence score is shown
    - A probability distribution chart is rendered
    - A Grad-CAM heatmap overlay is visible
11. Try **Compare All** to run ensemble inference across all 6 models

---

## 19. Troubleshooting Common Issues

### 🔴 `git lfs pull` fails or `.pth` files are tiny

**Symptom:** Model weight files in `training_pipeline/models/` are ~130 bytes instead of megabytes.

**Solution:**

```bash
git lfs install
git lfs pull
```

If bandwidth is limited, pull individual files:

```bash
git lfs pull --include="training_pipeline/models/efficientnet_b3_best.pth"
```

---

### 🔴 `ModuleNotFoundError: No module named 'argon2'`

**Symptom:** Backend crashes on startup because `passlib` cannot find the Argon2 backend.

**Solution:**

```bash
pip install argon2-cffi
```

On Windows, if this fails, install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.

---

### 🔴 `CUDA out of memory`

**Symptom:** GPU runs out of VRAM during inference or training.

**Solutions:**
- The backend's LRU cache already limits concurrent models to 2. If still OOM, edit `app.py`:
  ```python
  model_cache = ModelCache(max_size=1)  # Keep only 1 model loaded
  ```
- For training, reduce batch size in `training_pipeline/train.py`:
  ```python
  BATCH_SIZE = 16  # or even 8
  ```
- Close other GPU-consuming applications.

---

### 🔴 `ConnectionRefusedError` when starting backend (MongoDB)

**Symptom:** `pymongo.errors.ServerSelectionTimeoutError` or connection refused to MongoDB.

**Solution:**
1. Verify MongoDB is running:
   ```bash
   # Windows
   Get-Service MongoDB
   
   # Linux/macOS
   sudo systemctl status mongod
   ```
2. Check your `MONGODB_URI` in `.env` matches your MongoDB setup.
3. For Atlas: ensure your IP is whitelisted in Network Access settings.

---

### 🔴 CORS errors in browser console

**Symptom:** Frontend at `localhost:3000` gets blocked by CORS when calling backend at `localhost:8000`.

**Solution:** The backend already allows `http://localhost:3000`. If you're using a different port, add it to the CORS origins in `app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:YOUR_PORT"],
    ...
)
```

---

### 🔴 `[WARNING] pytorch-grad-cam not installed`

**Symptom:** Backend starts but Grad-CAM is disabled.

**Solution:**

```bash
pip install grad-cam
```

---

### 🔴 PowerShell execution policy blocks `venv\Scripts\Activate.ps1`

**Symptom:** Cannot activate virtual environment on Windows PowerShell.

**Solution:**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### 🔴 `npm install` fails with dependency conflicts

**Symptom:** Peer dependency resolution errors during frontend install.

**Solution:**

```bash
npm install --legacy-peer-deps
```

---

### 🔴 Frontend shows blank page or API errors

**Symptom:** Frontend loads but shows errors when trying to fetch data.

**Checklist:**
1. Is the backend running on port 8000? (`python app.py`)
2. Is `API_BASE` in `frontend/src/lib/auth.ts` set to `http://localhost:8000`?
3. Check browser DevTools → Console and Network tabs for specific error messages.
4. Verify the `.env` file exists and has valid `MONGODB_URI`.

---

### 🔴 Training script fails with path errors

**Symptom:** `FileNotFoundError` when running `training_pipeline/train.py`.

**Solution:** The training script has hardcoded absolute paths. Update them:

```python
# In training_pipeline/train.py, change these lines:
CSV_PATH = r"path/to/your/Dataset/train.csv"
IMAGE_DIR = r"path/to/your/Dataset/colored_images"
```

---

### 🔴 `torch.cuda.amp.autocast` deprecation warning

**Symptom:** Warning messages during training about deprecated AMP API.

**Note:** This is a non-blocking warning. The training will complete successfully. To suppress it, update `train.py` to use the new API:

```python
# Replace:
from torch.cuda.amp import autocast, GradScaler
# With:
from torch.amp import autocast, GradScaler
```

---

## Quick Start Summary

For developers who just want to get up and running quickly:

```bash
# 1. Clone and pull model weights
git clone https://github.com/yourusername/diabetic-retinopathy.git
cd diabetic-retinopathy
git lfs pull

# 2. Backend setup
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows
# source venv/bin/activate            # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure environment
# Create .env file with MONGODB_URI, DB_NAME, JWT_SECRET_KEY
# (See Step 9 above for details)

# 4. Start backend (Terminal 1)
python app.py

# 5. Frontend setup and start (Terminal 2)
cd frontend
npm install
npm run dev

# 6. Open browser → http://localhost:3000
```

---

*This guide corresponds to the requirements documented in [REQUIREMENTS.md](./REQUIREMENTS.md). For a complete list of all dependencies and their versions, refer to that document.*
