# OhMyGig AI
### *"Predict. Protect. Pay."*
> AI-Powered Parametric Income Insurance for India's Gig Delivery Workers

![Hackathon](https://img.shields.io/badge/Guidewire-DEVTrails%202026-purple)
![Platform](https://img.shields.io/badge/Platform-Flutter%20Mobile--First-blue)
![AI](https://img.shields.io/badge/AI-RF%20%7C%20XGBoost%20%7C%20TabNet%20%7C%20LSTM%20%7C%20GNN-green)

---

## 📌 Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [Persona — Rahul, Hyderabad](#3-persona--rahul-hyderabad)
4. [System Architecture](#4-system-architecture)
5. [Multi-Source Data Layer](#5-multi-source-data-layer)
6. [AI & ML Integration](#6-ai--ml-integration)
7. [Actuarial Premium Model](#7-actuarial-premium-model)
8. [Parametric Triggers](#8-parametric-triggers)
9. [Insurance Economic Model](#9-insurance-economic-model)
10. [Proportional Payout Logic](#10-proportional-payout-logic)
11. [Edge Cases & System Resilience](#11-edge-cases--system-resilience)
12. [Adversarial Defense & Anti-Spoofing Strategy](#12-adversarial-defense--anti-spoofing-strategy)
13. [User Workflow](#13-user-workflow)
14. [Tech Stack](#14-tech-stack)
15. [Analytics Dashboard](#15-analytics-dashboard)
16. [Business Model](#18-business-model)


---

## 1. Problem Statement

India's ~12 million platform-based gig delivery workers (Zomato, Swiggy, Blinkit, Zepto) survive on daily income with **zero financial protection** against external disruptions.

| Disruption | Frequency in Hyderabad | Monthly Income Impact |
|---|---|---|
| Extreme Heat (>42°C) | March–June, 18–22 days | ₹2,000–₹3,500 lost |
| Heavy Rain + Flash Flood | June–Oct, sudden spikes | ₹1,500–₹4,000 lost |
| Severe Pollution (AQI ≥300) | Oct–Jan | ₹500–₹1,500 lost |

**Structural gap:** Health, accident, and vehicle insurance exist. **Income loss protection does not.**

Workers lose **20–30% of monthly earnings** with no safety net. OhMyGig AI closes this gap.

---

## 2. Solution Overview

**OhMyGig AI** is a mobile-first, actuarially priced, AI-driven parametric income protection platform for food delivery workers in Hyderabad.

| Pillar | What It Does |
|---|---|
| 🔮 **Predict** | ML ensemble (RF + XGBoost + TabNet) forecasts disruption probability and prices premiums dynamically |
| 🛡️ **Protect** | Multi-source triggers monitor conditions every 30 mins — zero manual input required |
| 💸 **Pay** | Multi-signal fraud validation → pre-agreed UPI payout in under 30 seconds |

> Most platforms do: `weather → payout`
> OhMyGig does: `multi-source verification → AI risk scoring → actuarial pricing → fraud validation → instant payout`

---

## 3. Persona — Rahul, Hyderabad

| Field | Detail |
|---|---|
| Age / Location | 28, Kukatpally, Hyderabad (flood-prone zone) |
| Platform | Zomato(primary), Swiggy (secondary) |
| Daily Earnings | ₹800–₹1,000/day (verified via Zomato/Swiggy Earnings API) |
| Weekly Target | ₹5,600–₹7,000 |
| Peak Hours | 12–3 PM (lunch), 7–10 PM (dinner) |
| Annual Unprotected Loss | ₹7,000–₹15,000 |
| Annual Premium (Standard) | ~₹5,250 |

**Why Rahul buys OhMyGig AI:**
- ₹14/day auto-deducted from Zomato/Swiggy earnings — he never manually pays
- No paperwork, no claim filing — he just works
- First month FREE — he experiences a real payout before paying anything
- Payout hits the same UPI Zomato/Swiggy uses: instant trust

---

## 4. System Architecture

```
        [Zomato/Swiggy Partner App — OhMyGig Module]
                          ↓
              API Gateway (FastAPI + JWT Auth)
                          ↓
          Core Backend (Firebase Firestore + PostgreSQL)
                          ↓
┌────────────────────────────────────────────────────────┐
│  ML Risk Engine       (Ensemble: RF + XGBoost + TabNet)│
│  Premium Engine       (Actuarial + Zone + Season)      │
│  Trigger Engine       (Multi-source fusion, 30-min)    │
│  Behavioral Engine    (Activity Signal Pipeline)       │
│  Fraud Engine         (CVS + LSTM + GNN + Isolation F) │
└────────────────────────────────────────────────────────┘
                          ↓
          Claim Decision Engine (CVS threshold routing)
                          ↓
          Payment Engine (Razorpay Sandbox / Mock UPI)
                          ↓
     Zomato/Swiggy Settlement Adjustment (earnings net of payout)
                          ↓
         Analytics Dashboard (Worker + Admin + Insurer)
```

Each module is independently deployable and stateless.

---

## 5. Multi-Source Data Layer

Single-API reliance creates false triggers and missed events. OhMyGig AI uses **5-source cross-validated fusion** for maximum accuracy.

| Source | Data | Usage |
|---|---|---|
| **OWM One Call 3.0** | Rainfall mm/3hr (1km² grid), Temp, Wind | Primary trigger evaluation |
| **WAQI API** | Real-time AQI, PM2.5, PM10, NO2 | Pollution trigger |
| **IMD (India Met Dept)** | Official flood/storm alerts, red/orange warnings | Flood zone validation |
| **NOAA Open Data** | Historical Hyderabad rainfall & temperature normals | ML training data |
| **Device Sensors** | GPS, accelerometer, gyroscope | CVS fraud scoring |

**BASIS RISK FIX (Critical):**
- OWM One Call 3.0 provides **1km × 1km grid** — not city-level station data
- Rahul's live GPS delivery zone is cross-matched to the 1km² grid cell
- Trigger fires **only** if his current GPS zone is within the disruption zone
- LSTM verifies signal is sustained — not a 5-minute false spike

**Data Fusion Logic:**
```python
def fuse_disruption_signals(lat, lon, zone_id, worker_gps_zone):
    rain_owm = get_1km_rainfall_owm(lat, lon)       # 1km² grid
    imd_alert = get_imd_alert(zone_id)              # official alert
    temp = get_temperature_owm(lat, lon)
    aqi = get_aqi_waqi(lat, lon)
    flood_flag = check_flood_zone_db(zone_id)
    worker_in_zone = (worker_gps_zone == zone_id)   # basis risk fix

    if not worker_in_zone:
        return NO_TRIGGER  # Worker not in disrupted zone = no payout

    rain_confidence = 1.0 if (rain_owm > 20 and imd_alert) \
                 else 0.85 if rain_owm > 20 \
                 else 0.70 if imd_alert \
                 else 0.0

    lstm_confidence = lstm_model.predict(last_6_readings)
    final_confidence = (rain_confidence + lstm_confidence) / 2

    if final_confidence >= 0.80:
        return TRIGGER_CONFIRMED
    return NO_TRIGGER
```

If OWM shows 18mm (below threshold) but IMD has issued a flood advisory — the system still triggers. This prevents missed payouts during genuine disasters.

**API Cost at scale:** OWM One Call 3.0 ~₹8,000/month for 1,000 users polling every 30 min — built into operating cost model.

---

## 6. AI & ML Integration

### 6.1 Risk Prediction Engine (runs at policy purchase)

**Ensemble Stack:**
- **Layer 1 — Random Forest:** `P_trigger` base probability (interpretable, auditable)
- **Layer 2 — XGBoost:** Zone-specific non-linear refinement
- **Layer 3 — TabNet (DL):** Complex feature interaction capture
- **Output:** Weighted ensemble → Final `P_trigger` (0.0–1.0)

**Features:**
- Historical rainfall mm (52 weeks, zone-specific 1km² grid)
- Max temperature per week
- AQI weekly average
- Flood zone flag (binary, IMD database)
- Month / season encoding
- Zone cluster ID (Kukatpally, HITEC City, Gachibowli, etc.)
- IMD alert frequency in zone (past 1 year)
- Verified income tier (Zomato/Swiggy Earnings API)

**ML runs ONCE at purchase. NOT at claim time.** Output feeds directly into the actuarial formula → premium locked.

**Example:**
```
Zone: Kukatpally | Season: Peak Monsoon
Historical flooding: HIGH | Rainfall forecast: MODERATE
→ P_trigger = 0.42
```

### 6.2 Trigger Verification Engine (runs every 30 min)

LSTM time-series model:
- Looks at last 6 data points (3 hours of history)
- Determines: "Is this a sustained event or a false API spike?"
- Confidence ≥ 0.80 required before trigger confirmed
- Prevents: brief false spikes, sensor noise, API glitches

### 6.3 Fraud Detection Engine (runs at claim time)

Multi-layer CVS scoring + Isolation Forest + LSTM Autoencoder + GNN — see Section 12 for full detail.

---

## 7. Actuarial Premium Model — Full Detailed Version

This is OhMyGig AI's strongest technical differentiator. The formula operates in **weekly terms**, producing an industry-standard **Loss Ratio of 55–65%**.

### Formula

```
Premium_weekly = (EL × Ef × RL) / (1 − er) + Fixed

Where: EL = P_trigger × Dw × S × min(3, d_expected)
```

### Symbol Table

| Symbol | Meaning | Value |
|---|---|---|
| `P_trigger` | ML ensemble output — probability of ≥1 disruption event in zone this week (RF + XGBoost + TabNet) | 0.0–1.0 |
| `Dw` | Verified weekly income (Zomato/Swiggy Earnings API, not self-reported) | ₹5,600–₹7,000 |
| `S` | Severity factor — partial income replacement, not full (workers still earn something during disruption) | 0.80 |
| `min(3, d)` | Expected disruption days this week, capped at 3 to prevent runaway EL in extreme zones | ≤ 3 |
| `Ef` | Exposure factor — fraction of weekly income we are actually pricing against | 0.14 |
| `RL` | Risk load — regulatory capital buffer + model variance cushion | 1.15 |
| `er` | Expense ratio + profit margin | 0.22 (12% ops + 10% margin) |
| `Fixed` | Flat cost per policy per week (infra, OWM API, FCM, admin) | ₹10 |

### Fully Substituted Formula

```
Premium_weekly = ((P_trigger × Dw × 0.80 × 3) × 0.14 × 1.15) / 0.78 + 10

Daily Premium  = Weekly Premium ÷ 7 active days
```

---

### Pricing Tiers (Daily, ML-Dynamic, Auto-Deducted)

NOT weekly. NOT flat. **DAILY and SEASONAL** — priced by ML per zone.

| Plan | Off-Season | Pre-Monsoon | Peak Monsoon | Pollution | Annual Cost |
|---|---|---|---|---|---|
| Starter | ₹7/day | ₹10/day | ₹14/day | ₹8/day | ~₹3,500 |
| Standard | ₹14/day | ₹18/day | ₹25/day | ₹16/day | ~₹5,250 |
| Pro | ₹21/day | ₹28/day | ₹35/day | ₹24/day | ~₹7,500 |

**Why daily beats weekly:**
- ₹14/day = 1.5% of ₹900 daily income (vs 3–5% urban average — we're CHEAP)
- Deducted automatically via Zomato/Swiggy settlement — Rahul never manually pays
- Zero premium on non-working days — fair and efficient
- Seasonal surge: premium rises in July when Rahul KNOWS he needs it

**Key principle:** The ML does not price each worker to exact breakeven. It calibrates the **pool** so the aggregate Loss Ratio hits **55–65%** across all zones. Low-risk zones overpay relative to EL → cross-subsidise high-risk zones. Identical to how health insurance and all global parametric pools operate.

---

## 8. Parametric Triggers

Monitored automatically every **30 minutes** between **8 AM – 10 PM IST** — delivery hours only (prevents off-hours fraud).

| Trigger | Condition | Sources | Payout (Baseline) | Cooldown |
|---|---|---|---|---|
| Heavy Rain | >20mm / 3hr | OWM 1km² + IMD | ₹250 | 6 hrs |
| Extreme Heat | ≥42°C sustained | OWM + NOAA baseline | ₹250 | 24 hrs |
| Severe Pollution | AQI ≥300 | WAQI + CPCB | ₹175 | 12 hrs |
| Flash Flood | Rain >20mm + flood zone match | OWM + IMD alert + Zone DB | ₹500 | 24 hrs |
| Combined Bonus | ≥2 triggers simultaneously | Internal logic | +₹150 stacks | Per event |

- **LSTM Confidence Gate:** trigger only fires if confidence ≥ 0.80
- **GPS Zone Match Gate:** trigger only fires if worker is in the disrupted zone

Cooldown logic prevents duplicate payouts for the same sustained event.

---

## 9. Insurance Economic Model

### Pre-Agreed Payout Table (Locked at Purchase)

**PRINCIPLE:** Payout amount is PRE-AGREED and locked at policy purchase. When trigger fires → pay. ZERO calculation at claim time. (Same model as Arbol, Swiss Re, Raincoat, Jumpstart globally.)

Baseline — Secunderabad, Medium Risk — Standard Plan (₹14/day):

| Trigger | Payout | vs Daily Premium | vs Annual |
|---|---|---|---|
| Heavy Rain (>20mm/3hr) | ₹250 | +₹236 ✅ | 14× daily |
| Extreme Heat (≥42°C) | ₹250 | +₹236 ✅ | 14× daily |
| Severe Pollution AQI≥300 | ₹175 | +₹161 ✅ | 10× daily |
| Flash Flood | ₹500 | +₹486 ✅ | 29× daily |
| Combined Bonus (≥2 types) | +₹150 | stacks ✅ | — |
| Weekly cap | ₹900 | 6× weekly premium | — |

**Zone multipliers (applied by ML at purchase):**

| Zone | Multiplier | Logic |
|---|---|---|
| HIGH risk | × 0.67 | More frequent events → lower per-event payout |
| MEDIUM | × 0.90 | Baseline |
| LOW risk | × 1.30 | Fewer events → higher per-event payout |
| HIGH-MEDIUM | × 0.80 | — |

**NET RESULT:** All zones converge to 55–60% loss ratio. Actuarially fair. Worker in every zone: every single trigger returns > their daily premium ✅

### Pool Sustainability

Standard Plan, 1,000 users, Hyderabad:

| Metric | Value |
|---|---|
| Monthly premium intake | ₹4,30,000 (₹14/day × 250 working days / 12) |
| Monthly claims (60% LR) | ₹2,58,000 |
| Monthly reserve | ₹1,72,000 ✅ |
| Loss Ratio target | 55–60% (industry standard parametric) |
| Loss Ratio at scale (1M+) | Converges to 62–65% as claims data matures |

### Reinsurance — City-Wide Catastrophe Protection

**SCENARIO:** Hyderabad flood. ALL 50K workers trigger simultaneously.
50K × ₹500 = ₹2.5 crore payout in ONE DAY. Monthly pool = ₹2.1 crore. **WITHOUT REINSURANCE = BANKRUPT.**

**SOLUTION: XL (Excess-of-Loss) Reinsurance via New India Assurance:**
- Trigger: >30% of users claim within any 24-hour window
- Reinsurance layer absorbs everything above our retention limit
- Munich Re / Swiss Re provides catastrophe reinsurance globally
- New India Assurance handles this in our licensed partner model

---

## 10. Proportional Payout Logic

OhMyGig AI does **not** issue flat daily payouts. Payouts are proportional to the actual disruption window — protecting pool sustainability while being fair to every worker.

```
Payout = max(Floor, (disruption_hours / 14) × daily_plan_payout)
```

**How the window works — Rahul (Standard Plan):**
```
Works 8 AM–12 PM normally
Rain triggers at 12:15 PM → clears at 2:20 PM
Disruption window = 2.05 hours
Payout = max(₹100, (2/14) × ₹350) = ₹100 (floor applied)
Rain stops → Rahul resumes deliveries normally
Coverage remains active for the rest of the day
```

**Payout scale (Standard plan):**

| Disruption Duration | Payout |
|---|---|
| <1 hour | ₹100 (floor) |
| 1–3 hours | ₹100–₹175 |
| 3–6 hours | ₹175–₹350 |
| >6 hours | ₹350 (full daily cap) |

**Between-event cooldown:** 6 hours minimum — Rahul can claim again if a second separate disruption occurs later in the same day.

---

## 11. Edge Cases & System Resilience

OhMyGig AI explicitly handles every real-world operational scenario:

| # | Scenario | System Response |
|---|---|---|
| 1 | Worker in different zone when trigger fires | GPS zone ≠ disruption zone → NO TRIGGER (basis risk fix) |
| 2 | Worker inactive >3hrs before trigger | Zomato/Swiggy API: no active orders → B-score = 0 → CVS drops → blocked |
| 3 | Back-to-back rain (stop + resume) | 6-hour cooldown → same event. No double payout |
| 4 | City-wide flood (>30% claim) | XL Reinsurance activates → New India Assurance absorbs excess |
| 5 | Worker logs off early (moral hazard) | Trigger = weather, not activity. Gets paid regardless = no gaming incentive |
| 6 | Disruption during free month | Full payout honoured — first payout = permanent trust acquisition |
| 7 | Phone dies mid-disruption | Claim auto-initiated **server-side** — app closed never blocks payout |
| 8 | GPS degrades in flood zone | CVS thresholds relax 15%; GPS jump penalties reduced |
| 9 | Part-time worker, off that day | Coverage = active days only. Zero premium = zero coverage (ESIC precedent) |
| 10 | New user — ML cold start | Platform cohort baseline 30 days → switch to personal LSTM baseline. Claims >₹100 need liveness check first 30 days |
| 11 | 300 genuine workers same event | Cluster detection fires BUT confirmed trigger = fraud flag removed. All claims processed normally |
| 12 | UPI payment fails | 3 auto-retries (5-min intervals) → "payment pending" hold. Payout never cancelled, only delayed |
| 13 | Seasonal worker, active 3 days/wk | Premium only on active days. Worker educated: "Coverage = delivery days" |
| 14 | Worker switches zone mid-week | Trigger uses current GPS zone. Premium recalculates at next renewal |
| 15 | Disruption on policy expiry day | Policy valid until midnight. 9 PM disruption = honoured |
| 16 | OWM API downtime | 3-source redundancy (OWM → WAQI → IMD). Failover within 60 sec. ALL 3 down = manual admin. SLA: max 2-hr delay, never denied |

---

## 12. Adversarial Defense & Anti-Spoofing Strategy

> **Context:** A 500-person GPS-spoofing syndicate organized via Telegram has drained competitor insurance pools. OhMyGig AI was designed from Day 1 with this exact attack in mind.

### Why GPS Alone Is Broken

FakeGPS apps spoof coordinates in under 30 seconds. **Any system relying solely on GPS is exploitable by design.**

> Authentic delivery activity leaves a rich, multi-dimensional behavioral fingerprint that GPS-spoofing alone cannot replicate.

### Genuine Worker vs. Bad Actor

| Signal | Rahul (stranded in rain) | Fraudster (at home) |
|---|---|---|
| GPS path | Natural route with organic variation | Straight-line or static FakeGPS path |
| Accelerometer | Constant road vibrations, engine idle | Completely flat — no motion |
| Movement today | 40–80 km across zone | 0–2 km |
| App session | 7–9 hrs foreground usage | No prior activity before claim |
| Device | Registered weeks/months ago | New device or emulator |
| Claim timing | Random, scattered | Synchronized with 499 others within 90 sec |

### Layer 1 — Rule Engine (Hard Blocks)

```python
if android.FLAG_MOCK_LOCATION:         block()   # FakeGPS detected
if no_gyroscope and battery == 100:    block()   # Emulator detected
if sim_age_days < 7:                   elevate_risk()
if Zomato_Swiggy_inactive_hours > 3:          block()   # Worker not working
```

### Layer 2 — Claim Validity Score (CVS)

```
CVS = (0.25 × L) + (0.30 × M) + (0.25 × B) + (0.20 × E)
```

| Component | Signal | Weight |
|---|---|---|
| L — Location | GPS + Cell Tower triangulation cross-reference | 25% |
| M — Movement | Gyroscope, accelerometer, route continuity | 30% |
| B — Behavioral | App session history, Zomato/Swiggy activity pre-claim | 25% |
| E — Environmental | Nearby verified workers' activity consistency | 20% |

Movement has the highest weight (30%) — physical sensor manipulation cannot be faked with a GPS app alone.

**Device Sensor Fusion:**
```python
if accelerometer_variance < STATIC_THRESHOLD:   fraud_score += 30
if gyroscope_delta_sum < MOTION_THRESHOLD:      fraud_score += 25
if route_smoothness > FAKE_GPS_SMOOTHNESS:      fraud_score += 20
if movement_distance_today < 5.0:               fraud_score += 35
if speed_variance < 0.5:                        fraud_score += 25
```

**Cell Tower Triangulation:** GPS can be faked. Cell tower registration cannot. If GPS says Kukatpally but tower shows Secunderabad, CVS drops sharply.

### Layer 3 — Isolation Forest (Per-User ML Anomaly)

```python
from sklearn.ensemble import IsolationForest

features = [movement_km, speed_variance, app_session_hours,
            claims_30d, zone_cluster_flag, time_since_last_claim,
            device_age_days, sensor_motion_score]

model = IsolationForest(contamination=0.05)
anomaly_score = model.decision_function([features])
# Negative score = behavioral outlier = fraud flag
```

### Layer 4 — LSTM Autoencoder (Personal Behavioral Baseline)

- Learns Rahul's personal baseline over 30 days (e.g., "7-9 hrs active, 40-70 km, peaks at lunch + dinner")
- Claim deviating from **personal** baseline → anomaly spike
- Cold start: uses platform cohort baseline until 30 days of data
- Accuracy: 94%+ fraud detection (2024 research benchmark)

### Layer 5 — Graph Neural Network (Fraud Ring Detection)

- Isolation Forest sees INDIVIDUAL users. GNN sees **RELATIONSHIPS** across users.
- 500-person Telegram ring: all nodes connected → GNN detects cluster
- Telegram sync detection: `if stddev(claim_timestamps) < 90 sec → ring flag`
- Used by Allianz, AXA, PayPal for fraud ring detection
- No other InsurTech hackathon team will have this

**Geographic Cluster Detection:**
```python
if count_claims(zone_id, window_minutes=15) > CLUSTER_THRESHOLD:
    if not confirmed_trigger_event(zone_id):
        flag_all_claims_in_cluster(zone_id)   # Fraud ring
    # Confirmed trigger → mass legitimate event, no flag
```

### Graduated Response Framework

**Principle: Delay, Never Deny.**

| CVS Score | Action | Worker Sees |
|---|---|---|
| >0.75 | ✅ Instant payout (<30 sec) | "₹250 credited — rain detected in your zone" |
| 0.50–0.75 | ⚠️ 90-sec delay, passive log | "Verifying — payout incoming" |
| 0.30–0.50 | 🔶 Liveness selfie check | "Quick confirm — payout releases in 2 min" |
| <0.30 | 🚨 Admin review queue | "Claim under review — expected: 2 hrs" |

**Weather-degraded zone flag:** During confirmed >20mm/3hr rain, CVS thresholds auto-relax 15% — protecting honest workers whose GPS drops in actual bad weather.

### Why This System Is Syndicate-Resistant

| Attack Vector | OhMyGig Defense |
|---|---|
| FakeGPS app | Sensor fusion detects static device; cell tower mismatch |
| Home-based claiming | Movement score flags zero physical activity |
| Emulator farming | Device integrity check blocks at registration |
| Telegram mass-claim | GNN cluster detection + timestamp sync analysis |
| New SIM / device | Device age flag + elevated initial risk scoring |
| GPS jump spoofing | Route continuity score detects unnatural paths |

> A fraudster who must physically move 5km, maintain 7+ hours of app history, use a genuine device, and claim at an uncoordinated time has essentially become a real delivery worker.

---

## 13. User Workflow

```
1.  Open Zomato/Swiggy Partner App → tap "OhMyGig Insurance" tab
2.  Zomato/Swiggy income data pre-filled (API verified — no manual entry)
3.  ML risk profile generated → zone risk shown on map
4.  Plan selected → daily premium shown (₹7/₹14/₹21)
5.  One tap → Coverage ACTIVE. First 30 days FREE.
6.  Every 30 min: multi-source weather data polled
7.  GPS zone cross-matched to weather grid (1km² basis risk fix)
8.  LSTM verifies signal is sustained (confidence ≥ 0.80)
9.  Trigger confirmed → pre-agreed payout released
10. Fraud engine: CVS score computed (all 5 layers)
11. Decision: Instant / Delay / Selfie / Review
12. Payout hits UPI (same ID Zomato/Swiggy uses for daily earnings)
13. Notification: "₹250 credited — rain detected in your zone"
14. Dashboard: "Earnings protected this week: ₹500"
15. Daily premium deducted from next settlement: "₹14 OhMyGig | ₹886 net"
```

---

## 14. Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Mobile (embedded) | Flutter (Dart) | Embeds into Zomato/Swiggy module, sensor access |
| API Gateway | FastAPI (Python) | Async, lightweight, ML-native |
| Database | Firebase Firestore | Real-time sync, zero server ops |
| ML Service | scikit-learn + PyTorch | RF/XGBoost + LSTM + GNN |
| Primary Weather | OWM One Call 3.0 | 1km² grid resolution |
| AQI | WAQI API | Real Hyderabad AQI data |
| Flood Alerts | IMD API / scraper | Official Indian weather advisories |
| Historical | NOAA Open Data | ML training baseline |
| Income Verify | Zomato/Swiggy Earnings API | Verified income, no self-reporting |
| Payments | Razorpay Test Mode | Realistic UPI sandbox |
| Notifications | Firebase Cloud Messaging | Free, Flutter-native |
| Hosting | Render.com (free tier) | ML service deployment |

---

## 15. Analytics Dashboard

**Worker View:**
- Coverage status (active / inactive / expiring)
- Earnings protected (this week + all-time)
- Trigger event log: date, type, duration, payout
- Upcoming zone risk forecast (ML-based, 7 days)
- Premium history (daily deductions)
- Renewal nudge if lapsing

**Admin / Insurer View:**
- Loss ratio gauge (live, target 55–65%)
- Fraud alert feed (CVS scores, flagged claims)
- Hyderabad zone claim heatmap (real-time)
- Fraud ring cluster alerts (GNN detection)
- Predictive disruption forecast (LSTM, 48 hrs)
- Pool health meter (reserve vs claims)
- Payment retry queue
- Reinsurance trigger monitor (% users claiming)

---

## 16. Business Model

| Stream | Description |
|---|---|
| Premium margin | 35–45% of collected premiums after claims and operations |
| Platform licensing | 20% of premium to Zomato/Swiggy (distribution fee) |
| Reinsurance premium | Portion passed to New India Assurance |
| Risk data (future) | Anonymized disruption data → municipal and logistics planning |
| B2B white-label | License OhMyGig engine to Swiggy, Blinkit, Dunzo |

**Partnerships:** New India Assurance (IRDAI licensed reinsurance), Zomato/Swiggy (B2B2C distribution), PMSBY (govt welfare complement)

**TAM:** 12M workers × ₹5,250/yr (Standard) = **₹63,000 crore annual premium market**
**SAM (Hyderabad launch):** 3.5 lakh workers × ₹5,250 = **₹1,837 crore**

**Zomato/Swiggy incentive (judge-proof):**
- Revenue: 1 lakh workers × ₹14/day × 250 days × 20% = **₹7 crore/year**
- Retention: reduces monsoon worker churn (saves ₹3-5K/replacement)
- Legal: Code on Social Security 2025 mandates platforms provide social protection → OhMyGig = their compliance checkbox

---

- 2-minute video — `[YouTube link here]`

