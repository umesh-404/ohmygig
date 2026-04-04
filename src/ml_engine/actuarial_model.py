import math

class ActuarialEngine:
    """
    OhMyGig Actuarial Risk & Premium Pricing Engine
    Calculates the dynamic premium required to sustain a 55-65% Loss Ratio pool.
    """
    def __init__(self):
        self.exposure_factor = 0.14  # Fraction of weekly income at risk
        self.risk_load = 1.15        # Buffer + model variance cushion
        self.expense_ratio = 0.22    # 12% ops + 10% margin
        self.fixed_cost = 10         # Flat infra/api cost per policy/week (INR)
        self.severity_factor = 0.80  # Partial income replacement, not full

    def calculate_weekly_premium(self, p_trigger: float, weekly_income: float, expected_disruption_days: int) -> float:
        """
        Calculates the weekly premium dynamically based on ML outputs for a specific zone.
        
        :param p_trigger: Probability of >=1 disruption event in the zone this week (0.0 - 1.0)
        :param weekly_income: Verified weekly income via aggregator API (Zomato/Swiggy)
        :param expected_disruption_days: Expected disrupted days, capped at 3
        :return: Required Weekly Premium in INR
        """
        capped_days = min(3, expected_disruption_days)
        
        # EL = Expected Loss
        expected_loss = p_trigger * weekly_income * self.severity_factor * capped_days
        
        # Formula: Premium_weekly = (EL * Ef * RL) / (1 - er) + Fixed
        numerator = expected_loss * self.exposure_factor * self.risk_load
        denominator = 1.0 - self.expense_ratio
        
        weekly_premium = (numerator / denominator) + self.fixed_cost
        return round(weekly_premium, 2)

    def calculate_daily_deduction(self, weekly_premium: float) -> float:
        """
        Flattens weekly premium to daily micro-deduction.
        """
        daily_premium = weekly_premium / 7.0
        return round(daily_premium, 2)

# --- Quick Test ---
if __name__ == "__main__":
    engine = ActuarialEngine()
    # High Risk Zone (Monsoon)
    # P_trigger = 0.42, Income = 6000, expected days = 2
    wp = engine.calculate_weekly_premium(0.42, 6000, 2)
    dp = engine.calculate_daily_deduction(wp)
    print(f"High Risk -> Weekly Premium: ₹{wp}, Daily Deduction: ₹{dp}")
    
    # Low Risk Zone (Winter)
    # P_trigger = 0.08, Income = 6000, expected days = 1
    wp_low = engine.calculate_weekly_premium(0.08, 6000, 1)
    dp_low = engine.calculate_daily_deduction(wp_low)
    print(f"Low Risk -> Weekly Premium: ₹{wp_low}, Daily Deduction: ₹{dp_low}")
