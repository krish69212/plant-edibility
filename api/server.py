from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from src.model import load_checkpoint
from torchvision import transforms

app = FastAPI(title="Plant Edibility Classifier API")

# CORS for local web usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float


# Globals loaded on startup
MODEL = None
CLASS_TO_IDX: Optional[Dict[str, int]] = None
INFER_TRANSFORM = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
async def load_model_on_startup() -> None:
    global MODEL, CLASS_TO_IDX, INFER_TRANSFORM
    ckpt_env = os.getenv("MODEL_CHECKPOINT", "models/plant_edibility_resnet18.pth")
    ckpt_arg = os.getenv("MODEL_CHECKPOINT_OVERRIDE")  # optional alternative
    ckpt_path = Path(ckpt_arg or ckpt_env)

    if not ckpt_path.exists():
        print(f"WARNING: checkpoint not found at {ckpt_path}. The /predict endpoint will return 503 until a model is available.")
        return

    model, class_to_idx, meta = load_checkpoint(ckpt_path, map_location=str(DEVICE))
    model.eval()
    MODEL = model.to(DEVICE)
    CLASS_TO_IDX = class_to_idx

    mean = meta.get("mean", [0.485, 0.456, 0.406])
    std = meta.get("std", [0.229, 0.224, 0.225])
    img_size = int(meta.get("img_size", 224))

    INFER_TRANSFORM = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    print(f"Loaded model from {ckpt_path} on device {DEVICE}")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)):
    global MODEL, CLASS_TO_IDX, INFER_TRANSFORM
    if MODEL is None or CLASS_TO_IDX is None or INFER_TRANSFORM is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Provide a valid checkpoint and restart the server.")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = INFER_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        conf, pred_idx = torch.max(probs, dim=0)

    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
    predicted_class_raw = idx_to_class[int(pred_idx.item())]
    predicted_class = predicted_class_raw.replace("_", " ")
    confidence = float(conf.item())

    return PredictionResponse(predicted_class=predicted_class, confidence=confidence)


# Serve frontend
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    if not WEB_DIR.exists():
        return HTMLResponse("<h3>Frontend not found. Ensure 'web/' directory exists.</h3>")
    index_path = WEB_DIR / "index.html"
    html = index_path.read_text(encoding="utf-8")
    return HTMLResponse(html) 