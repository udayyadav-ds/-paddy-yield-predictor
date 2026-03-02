# ══════════════════════════════════════════════════════════════
# app.py — Production FastAPI App for Render Deployment
# Paddy Yield Predictor
# ══════════════════════════════════════════════════════════════

import os
import json
import time
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("paddy-api")

# ── Load model artifacts ──────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

try:
    model        = joblib.load(os.path.join(BASE, "models", "paddy_model.pkl"))
    scaler       = joblib.load(os.path.join(BASE, "scalers", "scaler.pkl"))
    FEATURE_COLS = json.load(open(os.path.join(BASE, "feature_cols.json")))
    log.info(f"✅ Model loaded  | Features: {len(FEATURE_COLS)}")
except Exception as e:
    log.error(f"❌ Failed to load model: {e}")
    raise

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title       = "🌾 Paddy Yield Predictor API",
    description = "Predicts paddy yield (Kg) from farm, weather & soil inputs.",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ══════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════

class FarmInput(BaseModel):
    """Single farm prediction request — pass any known features."""
    features: dict

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "Hectares"              : 6,
                    "Seedrate(in Kg)"       : 150,
                    "DAP_20days"            : 240,
                    "Urea_40Days"           : 162.78,
                    "Potassh_50Days"        : 62.28,
                    "total_rainfall"        : 500.0,
                    "avg_humidity"          : 80.0,
                    "avg_max_temp"          : 32.0,
                    "total_fertilizer"      : 555.0
                }
            }
        }

class PredictionResponse(BaseModel):
    predicted_yield_kg : float
    unit               : str = "Kg"
    inference_ms       : float

class BatchResponse(BaseModel):
    predictions : list[float]
    count       : int
    unit        : str = "Kg"
    inference_ms: float


# ══════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════

def make_prediction(features: dict) -> float:
    """Build feature vector → scale → predict."""
    vals = [float(features.get(f, 0.0)) for f in FEATURE_COLS]
    arr  = np.array([vals], dtype=np.float32)
    arr  = scaler.transform(arr)
    return float(model.predict(arr)[0])


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
def root():
    return {
        "message"   : "🌾 Paddy Yield Predictor API",
        "docs"      : "/docs",
        "health"    : "/health",
        "predict"   : "POST /predict",
        "bulk"      : "POST /predict/bulk",
        "features"  : FEATURE_COLS
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "status"     : "ok",
        "model"      : type(model).__name__,
        "n_features" : len(FEATURE_COLS)
    }


@app.post("/predict",
          response_model=PredictionResponse,
          tags=["Prediction"])
def predict_single(req: FarmInput):
    """
    Predict yield for ONE farm.

    Pass a dict of feature → value pairs.
    Missing features default to 0.
    """
    t0 = time.perf_counter()
    try:
        pred = make_prediction(req.features)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    return PredictionResponse(
        predicted_yield_kg = round(pred, 2),
        inference_ms       = round((time.perf_counter() - t0) * 1000, 2)
    )


@app.post("/predict/batch",
          response_model=BatchResponse,
          tags=["Prediction"])
def predict_batch(requests: list[FarmInput]):
    """
    Predict yield for MULTIPLE farms at once.

    Send a list of feature dicts.
    """
    if len(requests) > 1000:
        raise HTTPException(400, "Max 1000 rows per batch request.")

    t0   = time.perf_counter()
    preds = []
    try:
        for req in requests:
            preds.append(round(make_prediction(req.features), 2))
    except Exception as e:
        raise HTTPException(500, f"Batch prediction failed: {e}")

    return BatchResponse(
        predictions  = preds,
        count        = len(preds),
        inference_ms = round((time.perf_counter() - t0) * 1000, 2)
    )


@app.post("/predict/csv",
          tags=["Prediction"])
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file → get predictions for every row.

    CSV must have columns matching the model features.
    Returns a JSON list of predictions.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files accepted.")

    t0 = time.perf_counter()
    try:
        contents = await file.read()
        df       = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Fill missing columns with 0
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        arr   = df[FEATURE_COLS].values.astype(np.float32)
        arr   = scaler.transform(arr)
        preds = model.predict(arr).tolist()

    except Exception as e:
        raise HTTPException(500, f"CSV prediction failed: {e}")

    return {
        "predictions" : [round(p, 2) for p in preds],
        "count"       : len(preds),
        "unit"        : "Kg",
        "inference_ms": round((time.perf_counter() - t0) * 1000, 2)
    }


@app.get("/features", tags=["Info"])
def get_features():
    """Returns the list of features the model expects."""
    return {
        "features"   : FEATURE_COLS,
        "n_features" : len(FEATURE_COLS),
        "tip"        : "Missing features default to 0 in predictions."
    }
