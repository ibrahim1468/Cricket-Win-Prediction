import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from features import feature_engineering  # <-- Import from your external file

@st.cache_resource
def load_model():
    return joblib.load("best_cricket_model.pkl")

@st.cache_resource
def load_model_features():
    return joblib.load("model_features.pkl")

model = load_model()
model_features = load_model_features()

def predict_win_prob(df):
    prob = model.predict_proba(df)[0][1] * 100
    return prob

st.set_page_config(page_title="🏏 Cricket Chase Win Predictor", layout="wide")
st.title("🏏 Cricket Chase Win Probability Predictor")
st.markdown(
    """
Predict the probability of winning the chase based on current match conditions.  
Use the sidebar to input match details and explore what-if scenarios.
"""
)

# Sidebar inputs
st.sidebar.header("Match Inputs")

# Input fields with basic checks — will loop till inputs are valid
while True:
    innings_runs = st.sidebar.number_input("Current Score", min_value=0, step=1, value=100)
    innings_wickets = st.sidebar.slider("Wickets Fallen", min_value=0, max_value=10, value=2)
    target_score = st.sidebar.number_input("Target Score", min_value=1, step=1, value=200)

    # Prevent invalid runs_remaining < 0 scenario by capping
    max_runs_remaining = max(target_score - innings_runs, 0)
    runs_remaining = st.sidebar.number_input(
        "Runs Remaining", min_value=0, max_value=max_runs_remaining, step=1, value=max_runs_remaining
    )

    balls_remaining = st.sidebar.number_input(
        "Balls Remaining", min_value=0, max_value=300, step=1, value=60
    )

    # Validation checks
    error_messages = []
    if innings_runs > target_score:
        error_messages.append("⚠️ Current Score cannot be greater than Target Score.")
    if balls_remaining > 300:
        error_messages.append("⚠️ Balls Remaining cannot exceed 300 (max 50 overs).")
    if runs_remaining != target_score - innings_runs:
        error_messages.append(
            f"⚠️ Runs Remaining ({runs_remaining}) should equal Target Score - Current Score ({target_score - innings_runs})."
        )
    if error_messages:
        for msg in error_messages:
            st.sidebar.error(msg)
        st.sidebar.info("Please fix the input values to continue.")
        st.stop()
    else:
        break  # Inputs are valid, proceed

# Prepare input dataframe
input_dict = {
    "Innings Runs": [innings_runs],
    "Innings Wickets": [innings_wickets],
    "Target Score": [target_score],
    "Runs to Get": [runs_remaining],
    "Balls Remaining": [balls_remaining],
}
input_df = pd.DataFrame(input_dict)
input_df = input_df.reindex(columns=model_features)

# Feature engineering
input_df_fe = feature_engineering(input_df)
rrr_value = round(input_df_fe["RRR"].iloc[0], 2)

# Prediction
win_prob = predict_win_prob(input_df)
lose_prob = 100 - win_prob

# Layout display
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Current Win Probability")
    st.markdown(f"<h1 style='color:#0077FF'>{win_prob:.2f}%</h1>", unsafe_allow_html=True)
    st.markdown("---")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [win_prob, lose_prob],
        labels=["Win", "Lose"],
        colors=["#0077FF", "#D3D3D3"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 14},
    )
    ax.axis("equal")
    st.pyplot(fig)

with col2:
    st.subheader("Match Summary 📊")
    st.markdown(
        f"""
- **Current Score:** {innings_runs} 🏏  
- **Wickets Fallen:** {innings_wickets} ⚠️  
- **Target Score:** {target_score} 🎯  
- **Runs Remaining:** {runs_remaining} 🔥  
- **Balls Remaining:** {balls_remaining} ⏱️  
- **Required Run Rate:** {rrr_value} runs per over  
"""
    )
    st.markdown("---")

# What-if scenario exploration
st.subheader("📈 Explore What-If Scenarios")

scenario_variable = st.selectbox(
    "Select variable to simulate changes:",
    options=["Balls Remaining", "Wickets Fallen", "Runs Remaining"],
)

# Define slider ranges safely
if scenario_variable == "Balls Remaining":
    var_min, var_max = 0, balls_remaining if balls_remaining > 0 else 60
elif scenario_variable == "Wickets Fallen":
    var_min, var_max = 0, 10
else:  # Runs Remaining
    var_min, var_max = 0, runs_remaining if runs_remaining > 0 else target_score - innings_runs

var_values = st.slider(
    f"Adjust {scenario_variable} range", min_value=var_min, max_value=var_max, value=(var_min, var_max)
)

x_vals = np.linspace(var_values[0], var_values[1], 50)
probs = []
for val in x_vals:
    test_input = {
        "Innings Runs": innings_runs,
        "Innings Wickets": innings_wickets,
        "Target Score": target_score,
        "Runs to Get": runs_remaining,
        "Balls Remaining": balls_remaining,
    }
    test_input[scenario_variable] = val
    test_df = pd.DataFrame([test_input])
    test_df = test_df.reindex(columns=model_features)
    probs.append(predict_win_prob(test_df))

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(x_vals, probs, color="#0077FF", lw=3)
ax2.set_title(f"Win Probability vs {scenario_variable}")
ax2.set_xlabel(scenario_variable)
ax2.set_ylabel("Win Probability (%)")
ax2.grid(True)
st.pyplot(fig2)

# Quick insights with encouraging messages
st.subheader("💡 Quick Insights")
if win_prob > 75:
    st.success("Strong position! Keep pushing 🏏🔥")
elif win_prob > 40:
    st.info("It's a competitive chase. Every ball counts!")
else:
    st.warning("Tough chase. The pressure is on! ⚠️")

st.markdown("---")
st.markdown("Made with ❤️ by a cricket fanatic & data scientist.")
