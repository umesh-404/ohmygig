from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import sys
import os

# Adjust path strictly for mock demonstration purposes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from ml_engine.fraud_detection import ClaimValidityEngine

app = FastAPI(
    title="OhMyGig API Gateway",
    description="Core backend powering the Actuarial and Fraud Engine endpoints.",
    version="1.0"
)

cvs_engine = ClaimValidityEngine()

class TelemetryPayload(BaseModel):
    worker_id: str
    zone_id: str
    is_mock_location_flag: bool
    gps_cell_tower_delta_km: float
    accelerometer_variance: float
    aggregator_app_inactive_hours: float
    syndicate_cluster_density: int

@app.get("/api")
def health_check():
    return {"status": "operational", "system": "OhMyGig PIPELINE Active"}

@app.post("/api/evaluate")
def evaluate_payout_claim(payload: TelemetryPayload, authorization: str = Header(None)):
    """
    Receives raw sensor/app telemetry from Flutter client, runs CVS verification.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized Agent")
        
    try:
        # Pass telemetry dictionary derived from Pydantic model into engine
        assessment = cvs_engine.evaluate_claim(payload.model_dump())
        return {
            "worker_id": payload.worker_id,
            "status": "PROCESSED",
            "fraud_engine_response": assessment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import random

@app.get("/api/metrics")
def get_metrics():
    return {
        "aggregate_loss_ratio": 61.2,
        "daily_premium_pool": 420000,
        "fraud_ring_intercepts": 1402,
        "active_devices": random.randint(12000, 15000),
        "payouts_completed": 850
    }

@app.get("/api/matrix")
def get_regional_matrix():
    # Return mock zonal weather data and threshold limits mapping to recharts
    return [
        {"zone": "Kukatpally", "rainLevel_mm": 45, "riskMultiplier": 1.2, "claims": 120},
        {"zone": "HITEC City", "rainLevel_mm": 80, "riskMultiplier": 2.5, "claims": 310},
        {"zone": "Gachibowli", "rainLevel_mm": 65, "riskMultiplier": 1.8, "claims": 204},
        {"zone": "Secunderabad", "rainLevel_mm": 20, "riskMultiplier": 1.0, "claims": 40},
        {"zone": "Madhapur", "rainLevel_mm": 55, "riskMultiplier": 1.5, "claims": 150}
    ]

@app.get("/api/fraud")
def get_fraud_logs():
    return [
        {"id": "CLM-900", "worker": "ZOM-110", "cvs_score": 0.12, "reason": "Mock Location Detected", "status": "DENIED"},
        {"id": "CLM-899", "worker": "SWG-881", "cvs_score": 0.45, "reason": "Accelerometer Flatline", "status": "LIVENESS_CHECK"},
        {"id": "CLM-898", "worker": "ZOM-732", "cvs_score": 0.35, "reason": "GNN Guild Association", "status": "ADMIN_REVIEW"},
        {"id": "CLM-897", "worker": "SWG-115", "cvs_score": 0.98, "reason": "Verified", "status": "INSTANT_PAYOUT"}
    ]

@app.get("/api/payouts")
def get_payout_logs():
    return [
        {"trx_id": "TXN_783112", "amount": 800, "gateway": "Razorpay", "time": "2 mins ago", "status": "SUCCESS"},
        {"trx_id": "TXN_783111", "amount": 1200, "gateway": "UPI", "time": "5 mins ago", "status": "SUCCESS"},
        {"trx_id": "TXN_783110", "amount": 450, "gateway": "Razorpay", "time": "12 mins ago", "status": "SUCCESS"},
        {"trx_id": "TXN_783109", "amount": 800, "gateway": "UPI", "time": "22 mins ago", "status": "PENDING_BANK"}
    ]

if __name__ == "__main__":
    import uvicorn
    # Mocking standard startup 
    print("Starting OhMyGig FastApi Gateway Instance...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
