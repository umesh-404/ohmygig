# Technical Requirements Document (TRD)
## OhMyGig AI — System Architecture & Engineering Specifications

### 1. System Architecture Overview
The OhMyGig platform is composed of independent, stateless, microservice-based modules that interface directly with the aggregator app (Zomato/Swiggy).

- **Mobile Client Environment:** Embedded Flutter module within the primary partner app.
- **API Gateway:** FastAPI serving JWT authenticated REST endpoints.
- **Core Persistence DB:** Firebase Firestore for user state and fast sync; PostgreSQL for transaction auditing.
- **Decision Matrix Components:**
  - ML Risk / Premium Engine
  - Trigger / Actuarial Engine
  - Behavioral Fraud Engine
- **Settlement Logic:** Integrates Razorpay Test Mode out to partner API gateways for UPI payment routing.

### 2. Multi-Source Data Pipeline
Single APIs fall victim to lag or localized inaccuracies. The data ingestion layer uses a cross-validated 5-source node:
- **OWM One Call 3.0:** 1km² geo-grid monitoring for rain (mm/3hr) and baseline temperature.
- **WAQI API:** Feeds live AQI matrix scores into the pollutant trigger handler.
- **IMD Data Pipelines:** Scrapes/polls Official Red/Orange warnings and validates structural flood threats.
- **NOAA Open Data:** Static reference table providing climate baselines (training dataset).
- **Device SDK (Sensors):** Scrapes live telemetry from device Gyroscope, Accelerometer, and GPS coordinates.

### 3. Machine Learning Integration Layer
#### 3.1 Premium / Initial Risk Profile Ensemble
Run ONLY at policy purchase time to generate a localized probability of payout (`P_trigger`).
- **Layer 1 (Random Forest):** Auditable baseline tree matching historical weather risk probabilities.
- **Layer 2 (XGBoost):** Refines probability with gradient boosting for non-linear zone specifics.
- **Layer 3 (TabNet - DL):** Maps complex, non-obvious temporal interactions via attention mechanisms.

#### 3.2 Time-Series Confidence Module
- **Model:** LSTM (Long Short-Term Memory).
- **Function:** Reads the last 3 hours (6 ticks) of API data to dismiss localized noise, enforcing a >= 0.80 sustained-state confidence before allowing trigger.

### 4. Behavioral & Fraud Eradication Engine (Adversarial Defense)
Designed specifically to block 500+ entity GPS-spoofing syndicates via a structured tiered defense matrix:
- **Hard Rules / Kernel Checks:** Polls `android.FLAG_MOCK_LOCATION` and active physical sensor availability. Disqualifies emulators instantly.
- **Claim Validity Score (CVS):**
  - **Location (25%):** Assesses distance delta between reported GPS coordinates vs. Cellular Triangulation base stations.
  - **Movement (30%):** Validates raw XYZ variance from the accelerometer against expected vehicular vibrations.
  - **Behavioral (25%):** Hooks aggregator API confirming active deliveries before trigger time.
  - **Environmental (20%):** Verifies corroborating local telemetry from peripheral real-world users in the radius.
- **Machine Learning Security:**
  - The **Isolation Forest** assigns negative heuristic scores to solitary outliers.
  - A **Graph Neural Network (GNN)** sweeps metadata timestamps looking for clustered synchronized API hits indicating a Telegram-coordinated attack.

### 5. Actuarial Formula Logic
The fundamental weekly formulation operating in the Premium Engine service is:
`Premium_weekly = (((P_trigger × Dw × 0.80 × 3) × 0.14 × 1.15) / 0.78) + 10`
*The server parses this dynamically prior to checkout, offering a flattened daily subtraction factor.*

### 6. Cloud Infrastructure & Hosting
- **ML Services:** Containerized deployments running Python (PyTorch, scikit-learn) via Render.com (free tier) for the prototype.
- **Notifications Engine:** Firebase Cloud Messaging (FCM) pushing standard JSON structs asynchronously to the Flutter layer.
- **Resilience Strategy:** If OWM API fails, seamless failover cascades down to IMD / WAQI metrics. System gracefully defaults if fully disconnected by caching local state to Firestore and polling upon re-connect.
