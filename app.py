def predict_win_prob(df):
    """
    Predict win probability using base model with softer calibration for wickets & RRR.
    Returns probability in % (1 to 99).
    """
    base_prob = model.predict_proba(df)[0][1] * 100  # Base ML output

    # --- Wickets penalty (very gentle curve) ---
    wickets_fallen = df["Innings Wickets"].iloc[0]
    wickets_remaining = 10 - wickets_fallen
    alpha = 0.8  # Further reduced for minimal penalty (was 1.1)
    wicket_factor = (wickets_remaining / 10) ** alpha
    prob = base_prob * wicket_factor

    # --- Run Rate Required penalty (only for high RRR) ---
    runs_remaining = df["Runs to Get"].iloc[0]
    balls_remaining = df["Balls Remaining"].iloc[0]

    if balls_remaining > 0:
        current_rrr = (runs_remaining / balls_remaining) * 6
        rrr_threshold = 8.0  # Higher threshold to avoid penalizing low RRRs
        if current_rrr > rrr_threshold:
            excess = current_rrr - rrr_threshold
            rr_penalty = 1 / (1 + np.exp(0.3 * excess - 1.5))  # Even softer slope
            prob *= rr_penalty

    # --- Remove logistic smoothing to avoid over-compression ---
    # Clamp to realistic limits
    return max(min(prob, 99), 1)
