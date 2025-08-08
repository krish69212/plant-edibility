# Plant Edibility Classifier

A machine learning application that classifies plant edibility from images using transfer learning.

**⚠️ Safety Disclaimer**: This is for educational purposes only. Do NOT rely on it for real-world edibility decisions.

## Quick Start

### 1. Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Prepare Data
```
data/
  train/
    edible/
      image1.jpg
      ...
    not_edible/
      image2.jpg
      ...
```

### 3. Train
```bash
python -m src.train --data-dir data --epochs 10 --batch-size 16
```

### 4. Serve
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### 5. Use
- Web UI: `http://localhost:8000/`
- API: `POST /predict` with image file

## Project Structure
```
plant-edibility/
├── src/           # Training code
├── api/           # FastAPI server
├── web/           # Frontend
├── models/        # Saved checkpoints
└── data/          # Your dataset
```

## API Endpoints
- `GET /health` - Health check
- `POST /predict` - Upload image, get prediction
- `GET /` - Web interface

## Training Options
- `--arch resnet18|resnet50`
- `--freeze` - Train only classifier head
- `--epochs 10` - Number of training epochs
- `--batch-size 16` - Batch size (reduce if OOM)

## Environment Variables
- `MODEL_CHECKPOINT` - Path to model file (default: `models/plant_edibility_resnet18.pth`) 
