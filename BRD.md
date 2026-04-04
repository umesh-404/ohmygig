# Business Requirements Document (BRD)
## OhMyGig AI — Parametric Income Insurance for Gig Workers

### 1. Executive Summary
OhMyGig AI is an AI-powered, parametric income protection platform designed specifically for the 12 million platform-based gig delivery workers in India. Moving away from traditional claims-based insurance, OhMyGig leverages a multi-source data fusion approach (weather APIs, IMD alerts, WAQI, and device sensors) to automatically trigger micro-payouts when external disruptions (extreme heat, heavy rain, pollution, floods) prevent workers from earning their daily wage.

### 2. Objectives
- **Financial Inclusion:** Provide the first line of defense for gig workers against climate-induced income loss.
- **Parametric Efficiency:** Achieve zero manual claims processing and eliminate basis risk through 1km² geo-fencing.
- **Loss Ratio Control:** Sustain an industry-standard loss ratio of 55–65% using ML-driven dynamic pricing.
- **Fraud Eradication:** Utilize a multi-tiered Fraud Engine to identify organized syndicates, FakeGPS, and spoofing to prevent pool depletion.

### 3. Target Audience / Persona
- **Primary Persona:** Rahul (28, Hyderabad)
- **Role:** Platform Gig Worker (Zomato / Swiggy)
- **Earnings:** ₹800–₹1,000/day
- **Pain Point:** Unprotected daily income. External disruptions cost him ₹7,000–₹15,000 annually.
- **Value Proposition:** ₹14/day auto-deducted premium for immediate, frictionless UPI payouts under 30 seconds when disrupted. 

### 4. Products & Services Required
- **Starter Plan (Off-season):** ₹7 - ₹14 / day.
- **Standard Plan (Baseline):** ₹14 - ₹25 / day.
- **Pro Plan (High Activity):** ₹21 - ₹35 / day.
*Note: Premium cost dynamically adjusts based on the season and real-time ML risk evaluations.*

### 5. Business & Operational Logic
1. **Dynamic Actuarial Pricing:** The platform uses an ensemble ML model (Random Forest, XGBoost, TabNet) to set pricing individually for every user's zone.
2. **Proportional Payouts:** Daily payouts are prorated against the disruption window length to prevent moral hazard (e.g., stopping work entirely for minor events).
3. **Trigger Cooldowns:** Minimum elapsed time (e.g., 6 hours for rain) before another claim can trigger, ensuring the payout pool isn't drained by minor flurries.
4. **Reinsurance Layer:** "Excess of Loss" parameters via New India Assurance shield the platform if >30% of users claim within 24 hours (Macro-events).

### 6. Value Proposition to Stakeholders
- **Workers:** Trust via instant payouts, zero claim filing, easy affordability.
- **Insurers / Reinsurers:** AI-curated risk pools, completely transparent fraud mitigation, and a massive untapped premium TAM (₹63,000 Cr/yr).
- **Delivery Platforms (Zomato/Swiggy):** Reduced churn/attrition during monsoons, and direct compliance with the Code on Social Security 2025.

### 7. Core Success Metrics
- **Automated Payout Rate:** 99% of valid claims triggered and executed without human review.
- **Fraud Block Rate:** 100% of simulator/FakeGPS activity detected by the CVS protocol.
- **Target Loss Ratio:** 60% average.

### 8. Competitive Differentiator (Crucial for Phase 2)
Unlike "GigShield" and simple weather-trigger competitors:
- We solve **Basis Risk** by isolating the trigger to the user's specific 1km² GPS zone.
- We utilize an aggressive, 5-layer **Adversarial Defense (Fraud Engine)** containing GNNs to bust GPS-spoofing syndicates.
- We price based on **Actuarial Science**, generating a realistic Premium vs. Payout curve, which proves business viability to VC evaluators.
