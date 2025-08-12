import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("cricket_model.pkl")

# Format selection
match_format = st.selectbox("Match format", ["T20", "ODI"])

# Inputs
target = st.number_input("Target score", min_value=1, step=1)
score = st.number_input("Current score", min_value=0, step=1)
overs = st.number_input("Overs completed", min_value=0.0, step=0.1)
wickets = st.number_input("Wickets lost", min_value=0, step=1)

# Function to calculate win probability
def predict_win_prob(target, score, overs, wickets, match_format):
    # Overs & balls calculation
    balls_bowled = int(overs) * 6 + round((overs % 1) * 10)
    balls_remaining = (20 if match_format == "T20" else 50) * 6 - balls_bowled

    runs_required = target - score
    current_rr = score / (balls_bowled / 6) if balls_bowled > 0 else 0
    required_rr = runs_required / (balls_remaining / 6) if balls_remaining > 0 else 0

    # Base ML model prediction (scaled 0-100)
    base_pred = model.predict_proba(
        pd.DataFrame([[target, score, overs, wickets]],
                     columns=["Target", "Score", "Overs", "Wickets"])
    )[0][1] * 100

    # Format-aware wicket penalty (gentler in T20)
    if match_format == "T20":
        wicket_penalty = 1 - (wickets * 0.06)  # -6% per wicket
    else:
        wicket_penalty = 1 - (wickets * 0.05)  # -5% per wicket

    wicket_penalty = max(wicket_penalty, 0.5)  # never drop below 50% effect

    # RRR penalty using soft logistic curve
    rr_diff = required_rr - current_rr
    if match_format == "T20":
        rr_penalty = 1 / (1 + np.exp(0.8 * rr_diff))  # gentle slope
    else:
        rr_penalty = 1 / (1 + np.exp(0.5 * rr_diff))  # even gentler

    # Blend penalties instead of multiplying fully
    final_pred = base_pred * (0.6 + 0.4 * wicket_penalty) * (0.7 + 0.3 * rr_penalty)

    # Clamp to [0, 100]
    final_pred = max(0, min(100, final_pred))

    return final_pred

# Predict button
if st.button("Predict Win Probability"):
    win_prob = predict_win_prob(target, score, overs, wickets, match_format)
    st.metric("Win Probability (%)", f"{win_prob:.2f}")
