# streamlit_cricket_app.py
import streamlit as st
import pandas as pd
import joblib

# ===========================
# 1. Load your trained model
# ===========================
MODEL_PATH = "model_clean_without_target.joblib"  # adjust path
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Cricket Chase Predictor", page_icon="🏏", layout="centered")
st.title("🏏 Cricket Match Win Probability Predictor")

# ===========================
# 2. User inputs
# ===========================
st.subheader("Enter Match Details")

venue = st.selectbox("Venue", options=[
    "SuperSport Park", "Dubai International Cricket Stadium", "Eden Gardens", "M. Chinnaswamy Stadium"
])  # Add all venues your model knows

innings = st.radio("Innings", options=[1, 2])

current_runs = st.number_input("Current Runs", min_value=0, value=0, step=1)
wickets_in_hand = st.number_input("Wickets in Hand", min_value=0, max_value=10, value=10, step=1)
overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

# Only ask for target in 2nd innings
if innings == 2:
    target_score = st.number_input("Target Score", min_value=1, value=1, step=1)
    if current_runs > target_score:
        st.error("Current runs cannot exceed target for second innings!")
        st.stop()

# ===========================
# 3. Prepare input DataFrame
# ===========================
input_data = pd.DataFrame({
    "Venue": [venue],
    "Current Score": [current_runs],
    "Wickets_in_Hand": [wickets_in_hand],
    "Overs Completed": [overs_completed],
    "Innings": [innings],
    # placeholders for model-required columns
    "CRR": [0],
    "RRR": [0],
    "Runs_to_Get": [0],
    "Balls_Remaining": [0],
    "Pressure": [0],
    "Home_Advantage": [0],
})

# ===========================
# 4. Calculate features dynamically
# ===========================
TOTAL_OVERS = 50
balls_bowled = overs_completed * 6
balls_remaining = (TOTAL_OVERS * 6) - balls_bowled

# Current Run Rate
if overs_completed > 0:
    crr = current_runs / overs_completed
else:
    crr = 0
input_data["CRR"] = [crr]

if innings == 1:
    projected_score = int(crr * TOTAL_OVERS)
    input_data["Projected_Score"] = [projected_score]
else:
    runs_to_get = target_score - current_runs
    rrr = (runs_to_get * 6) / balls_remaining if balls_remaining > 0 else 0
    pressure = runs_to_get / (wickets_in_hand + 1)
    input_data["Runs_to_Get"] = [runs_to_get]
    input_data["Balls_Remaining"] = [balls_remaining]
    input_data["RRR"] = [rrr]
    input_data["Pressure"] = [pressure]

# ===========================
# 5. Predict probability
# ===========================
if st.button("Predict Probability"):
    try:
        proba = model.predict_proba(input_data)[0][1]  # probability of successful chase
        st.success(f"✅ Probability of successful chase: {proba*100:.2f}%")
        if innings == 1:
            st.info(f"Projected Score at end of innings: {projected_score}")
    except Exception as e:
        st.error(f"Error predicting: {e}")
