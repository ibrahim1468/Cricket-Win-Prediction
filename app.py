import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Define feature engineering function (needed for model loading) ---
def feature_engineering(df):
    df = df.copy()
    df["Current Score"] = df["Innings Runs"]
    df["Wickets Remaining"] = 10 - df["Innings Wickets"]
    df["RRR"] = np.where(
        df["Balls Remaining"] > 0,
        df["Runs to Get"] / (df["Balls Remaining"] / 6),
        0,
    )
    return df

# --- Cache model loading ---
@st.cache_resource
def load_model():
    model = joblib.load("best_cricket_model.pkl")
    return model

@st.cache_resource
def load_model_features():
    features = joblib.load("model_features.pkl")
    return features

model = load_model()
model_features = load_model_features()

# --- Predict function ---
def predict_win_prob(df):
    prob = model.predict_proba(df)[0][1] * 100
    return prob

# --- Streamlit page config and title ---
st.set_page_config(page_title="🏏 Cricket Chase Win Predictor", layout="wide")
st.title("🏏 Cricket Chase Win Probability Predictor")
st.markdown(
    """
Predict the probability of winning the chase based on current match conditions.
Use the sidebar to input match details and explore what-if scenarios.
"""
)

# --- Sidebar inputs ---
st.sidebar.header("Match Inputs")

innings_runs = st.sidebar.number_input("Current Score", min_value=0, value=100, step=1)
innings_wickets = st.sidebar.slider("Wickets Fallen", min_value=0, max_value=10, value=2)
target_score = st.sidebar.number_input("Target Score", min_value=1, value=200, step=1)
runs_remaining = st.sidebar.number_input(
    "Runs Remaining", min_value=0, value=target_score - innings_runs, step=1
)
balls_remaining = st.sidebar.number_input(
    "Balls Remaining", min_value=0, max_value=300, value=60, step=1
)

# --- Input validations with friendly messages ---
error_flag = False

if innings_runs > target_score:
    st.sidebar.error(
        "Oops! Current Score cannot be greater than Target Score. Please correct it."
    )
    error_flag = True

if balls_remaining > 300:
    st.sidebar.error(
        "Whoa! Balls Remaining can't exceed 300 (max 50 overs). Please correct it."
    )
    error_flag = True

if runs_remaining != target_score - innings_runs:
    st.sidebar.warning(
        f"Note: Runs Remaining ({runs_remaining}) doesn't match Target - Current Score ({target_score - innings_runs}). Please double-check."
    )

if error_flag:
    st.info("Please fix the errors above to see the predictions.")
    st.stop()

# --- Prepare input DataFrame ---
input_dict = {
    "Innings Runs": [innings_runs],
    "Innings Wickets": [innings_wickets],
    "Target Score": [target_score],
    "Runs to Get": [runs_remaining],
    "Balls Remaining": [balls_remaining],
}

input_df = pd.DataFrame(input_dict)
input_df = input_df.reindex(columns=model_features)

# --- Feature engineering ---
input_df_fe = feature_engineering(input_df)
rrr_value = round(input_df_fe["RRR"].iloc[0], 2)

# --- Prediction ---
win_prob = predict_win_prob(input_df)
lose_prob = 100 - win_prob

# --- Layout ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Current Win Probability")
    st.markdown(f"<h1 style='color:#0077FF'>{win_prob:.2f}%</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Pie chart
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

# --- Interactive What-If Scenarios ---
st.subheader("📈 Explore What-If Scenarios")

scenario_variable = st.selectbox(
    "Select variable to simulate changes:",
    options=["Balls Remaining", "Wickets Fallen", "Runs Remaining"],
)

# Slider ranges based on input sanity
if scenario_variable == "Balls Remaining":
    var_min, var_max = 0, balls_remaining if balls_remaining > 0 else 60
elif scenario_variable == "Wickets Fallen":
    var_min, var_max = 0, 10
else:  # Runs Remaining
    var_min, var_max = 0, runs_remaining if runs_remaining > 0 else target_score - innings_runs

var_values = st.slider(
    f"Adjust {scenario_variable} range", min_value=var_min, max_value=var_max, value=(var_min, var_max)
)

# Generate predictions for scenario
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
    prob = predict_win_prob(test_df)
    probs.append(prob)

# Scenario plot
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(x_vals, probs, color="#0077FF", lw=3)
ax2.set_title(f"Win Probability vs {scenario_variable}")
ax2.set_xlabel(scenario_variable)
ax2.set_ylabel("Win Probability (%)")
ax2.grid(True)
st.pyplot(fig2)

# --- Quick insights ---
st.subheader("💡 Quick Insights")
if win_prob > 75:
    st.success("Strong position! Keep pushing 🏏🔥")
elif win_prob > 40:
    st.info("It's a competitive chase. Every ball counts!")
else:
    st.warning("Tough chase. The pressure is on! ⚠️")

# --- Footer ---
st.markdown(
    """
---
Made with ❤️ by a cricket fanatic & data scientist.
"""
)
