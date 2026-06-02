"""FastAPI Service: ML inference API with distributed deployment"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import time

app = FastAPI(title="ML Pipeline API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    predictions: List[float]
    latency_ms: float
    replica_id: int

inference_service = None

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction via distributed inference"""
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    start = time.time()

    try:
        X = np.array(request.features).reshape(1, -1)
        result = inference_service.predict(X)

        latency_ms = (time.time() - start) * 1000

        return {
            "predictions": result["predictions"],
            "latency_ms": latency_ms,
            "replica_id": result.get("replica_id")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Liveness probe (Kubernetes)"""
    if inference_service is None:
        return {"status": "not_ready"}

    return {"status": "healthy"}

@app.get("/cluster-status")
async def cluster_status():
    """Get distributed cluster status"""
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return inference_service.get_cluster_health()
