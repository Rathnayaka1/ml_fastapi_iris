from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, numpy as np, json, os

app = FastAPI(title="Iris Classifier API")

# Try to load model and metadata
model, MODEL_META = None, {}
if os.path.exists("model.pkl"):
    try:
        model = joblib.load("model.pkl")
        print("✅ model.pkl loaded successfully")
    except Exception as e:
        print("⚠️ Failed to load model.pkl:", e)

if os.path.exists("model_metadata.json"):
    try:
        with open("model_metadata.json", "r") as f:
            MODEL_META = json.load(f)
        print("✅ model_metadata.json loaded successfully")
    except Exception as e:
        print("⚠️ Failed to load model_metadata.json:", e)


# ----- Request & Response Schemas -----
class IrisInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float


# ----- Routes -----
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Iris Classifier API is running"}


@app.get("/model-info")
def model_info():
    if not MODEL_META:
        raise HTTPException(status_code=500, detail="Model metadata not found. Train the model first.")
    return MODEL_META


@app.post("/predict", response_model=PredictionOutput)
def predict(inp: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Run train_model.py or copy model.pkl here.")

    try:
        features = np.array([[inp.SepalLengthCm, inp.SepalWidthCm,
                              inp.PetalLengthCm, inp.PetalWidthCm]])
        pred = model.predict(features)[0]
        proba = model.predict_proba(features).max()
        return {"prediction": pred, "confidence": float(proba)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))