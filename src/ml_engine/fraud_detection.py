import json
import logging

logger = logging.getLogger("OhMyGig.FraudEngine")

class ClaimValidityEngine:
    """
    Evaluates claims to intercept FakeGPS spoofing and structured syndicate attacks.
    Assigns a Claim Validity Score (CVS) taking into account 4 telemetry dimensions.
    """
    
    def __init__(self):
        # Weights for the 4 pillars of verification
        self.weights = {
            "L_location": 0.25,
            "M_movement": 0.30,
            "B_behavioral": 0.25,
            "E_environmental": 0.20
        }
    
    def evaluate_claim(self, telemetry_payload: dict) -> dict:
        """
        Calculates the CVS score. 
        Returns an action: Instant, Delay, Selfie_Check, or Admin_Review.
        """
        score_L = self._score_location(telemetry_payload)
        score_M = self._score_movement(telemetry_payload)
        score_B = self._score_behavioral(telemetry_payload)
        score_E = self._score_environmental(telemetry_payload)
        
        cvs = (score_L * self.weights["L_location"]) + \
              (score_M * self.weights["M_movement"]) + \
              (score_B * self.weights["B_behavioral"]) + \
              (score_E * self.weights["E_environmental"])
              
        action = self._determine_action(cvs)
        
        logger.info(f"Evaluated Claim -> CVS: {cvs:.2f} | Action: {action}")
        return {"cvs_score": round(cvs, 3), "action_required": action}

    def _score_location(self, data) -> float:
        # Cross reference GPS coordinates with cell tower triangulation
        if data.get('is_mock_location_flag', False):
            return 0.0 # Instant fail
        
        gps_cell_delta_km = data.get('gps_cell_tower_delta_km', 10.0)
        return 1.0 if gps_cell_delta_km < 2.0 else 0.4

    def _score_movement(self, data) -> float:
        # Heaviest weight. Checks accelerometer variance for real-world vehicular vibration
        accel_variance = data.get('accelerometer_variance', 0)
        if accel_variance < 0.5: # Device is sitting completely flat/static on a desk
            return 0.0
        return 1.0

    def _score_behavioral(self, data) -> float:
        # Checks if user was actively on a delivery app prior to claim
        inactive_hours = data.get('aggregator_app_inactive_hours', 10)
        return 1.0 if inactive_hours <= 1.0 else 0.0

    def _score_environmental(self, data) -> float:
        # Compares with localized genuine nodes (Isolation Forest & GNN mock layer)
        cluster_density = data.get('syndicate_cluster_density', 0)
        if cluster_density > 50: # 50 claims triggered at same millisecond = Telegram syndicate attack
            return 0.0
        return 0.9

    def _determine_action(self, cvs: float) -> str:
        if cvs > 0.75:
            return "INSTANT_PAYOUT"
        elif 0.50 < cvs <= 0.75:
            return "90_SEC_DELAY"
        elif 0.30 <= cvs <= 0.50:
            return "LIVENESS_SELFIE_REQUIRED"
        else:
            return "ADMIN_REVIEW"

# --- Quick Test ---
if __name__ == "__main__":
    engine = ClaimValidityEngine()
    
    # 1. Genuine Delivery Worker caught in rain
    genuine_payload = {
        "is_mock_location_flag": False,
        "gps_cell_tower_delta_km": 0.5,
        "accelerometer_variance": 5.4, # Moving bike
        "aggregator_app_inactive_hours": 0.1, # Just did a delivery
        "syndicate_cluster_density": 2
    }
    print("Genuine User:", engine.evaluate_claim(genuine_payload))
    
    # 2. Guy sitting at home using FakeGPS app
    fraud_payload = {
        "is_mock_location_flag": False, # Masked mock location
        "gps_cell_tower_delta_km": 15.0, # Tower says Secunderabad, GPS says Kukatpally
        "accelerometer_variance": 0.1, # Phone flat on desk
        "aggregator_app_inactive_hours": 8.0, # Not been working today
        "syndicate_cluster_density": 85 # Jumped into a telegram raid 
    }
    print("Fraud Syndicate Actor:", engine.evaluate_claim(fraud_payload))
