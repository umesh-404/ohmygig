from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import sys
import os

# Adjust path strictly for mock demonstration purposes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

@app.get("/")
def health_check():
    return {"status": "operational", "system": "OhMyGig Pipeline Active"}

@app.post("/claim/evaluate")
def evaluate_payout_claim(payload: TelemetryPayload, authorization: str = Header(None)):
    """
    Receives raw sensor/app telemetry from Flutter client, runs CVS verification.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized Agent")
        
    try:
        # Pass telemetry dictionary derived from Pydantic model into engine
        assessment = cvs_engine.evaluate_claim(payload.dict())
        return {
            "worker_id": payload.worker_id,
            "status": "PROCESSED",
            "fraud_engine_response": assessment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Mocking standard startup 
    print("Starting OhMyGig FastApi Gateway Instance...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
