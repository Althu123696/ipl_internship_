from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="IPL Runs Prediction", version="1.0")

# Load trained model once at startup
model = joblib.load("model/poly_linear_regression_pipeline.pkl")

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Custom metric for leakage gap
leakage_metric = Gauge("data_leakage_gap", "Gap between train and test R2")

def log_leakage_gap(train_score, test_score):
    leakage_metric.set(train_score - test_score)

class PlayerData(BaseModel):
    Inns: int
    NO: int
    Avg: float
    BF: int
    four_runs: int = Field(..., alias="4s")
    six_runs: int = Field(..., alias="6s")

@app.get("/")
def root():
    return {"message": "IPL API is running"}

@app.post("/predict")
def predict_runs(player: PlayerData):
    df = pd.DataFrame([player.dict(by_alias=True)])
    prediction = model.predict(df)[0]
    return {"Predicted Runs": round(float(prediction),0)}
