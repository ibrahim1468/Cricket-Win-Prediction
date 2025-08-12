import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from features import feature_engineering  # Your external feature engineering func

# --- Load model and artifacts ---
@st.cache_resource
def load_model():
    return joblib.load("best_cricket_model.pkl")

@st.cache_resource
def load_model_features():
    return joblib.load("model_features.pkl")

@st.cache_resource
def load_training_data():
    return pd.read_csv("col.csv")  # your training data to plot histogram

model = load_model()
model_features = load_model_features()
training_data = load_training_data()

# --- Predict function ---
def predict_win_prob(df):
    prob = model.predict_proba(df)[0][1] * 100
    return prob

# --- Page Setup ---
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

# Validation for runs_remaining min_value dynamically depends on target - innings_runs
runs_min = 0
runs_max = max(target_score - innings_runs, 0)
runs_remaining = st.sidebar.number_input(
    "Runs Remaining",
    min_value=runs_min,
    max_value=target_score,
    value=runs_max,
    step=1,
    help="Runs remaining to reach target (Target Score - Current Score)",
)

balls_remaining = st.sidebar.number_input(
    "Balls Remaining", min_value=0, max_value=300, value=60, step=1
)

# --- Input Validation & Friendly Messages ---
if innings_runs > target_score:
    st.sidebar.error("Oops! Current Score cannot be greater than Target Score.")
    st.warning("Please adjust the inputs above to keep the game logical.")
    st.stop()

if balls_remaining > 300:
    st.sidebar.error("Balls Remaining cannot exceed 300 (max 50 overs).")
    st.stop()

if runs_remaining != target_score - innings_runs:
    st.sidebar.warning(
        f"Runs Remaining ({runs_remaining}) does not match Target - Current Score ({target_score - innings_runs})."
    )

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

# Debug: Show engineered features
st.write("### Engineered Features Input to Model")
st.dataframe(input_df_fe)

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

    # Show feature importances (for tree models)
    st.subheader("Feature Importance")
    try:
        importances = model.named_steps["clf"].feature_importances_
        feat_imp_df = pd.DataFrame({
            "Feature": model_features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(feat_imp_df.set_index("Feature"))
    except AttributeError:
        st.info("Feature importance not available for this model type.")

# --- What-If Scenario Analysis ---
st.subheader("📈 Explore What-If Scenarios")

scenario_variable = st.selectbox(
    "Select variable to simulate changes:",
    options=["Balls Remaining", "Wickets Fallen", "Runs Remaining"],
)

# Define dynamic slider ranges
if scenario_variable == "Balls Remaining":
    var_min, var_max = 0, max(balls_remaining * 2, 60)
elif scenario_variable == "Wickets Fallen":
    var_min, var_max = 0, 10
else:  # Runs Remaining
    var_min, var_max = 0, max(runs_remaining * 2, target_score)

var_values = st.slider(
    f"Adjust {scenario_variable} range",
    min_value=var_min,
    max_value=var_max,
    value=(var_min, var_max),
)

# Generate values and predict
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

# Plot scenario results
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(x_vals, probs, color="#0077FF", lw=3)
ax2.set_title(f"Win Probability vs {scenario_variable}")
ax2.set_xlabel(scenario_variable)
ax2.set_ylabel("Win Probability (%)")
ax2.grid(True)
st.pyplot(fig2)

# --- Quick Insights ---
st.subheader("💡 Quick Insights")
if win_prob > 75:
    st.success("Strong position! Keep pushing 🏏🔥")
elif win_prob > 40:
    st.info("It's a competitive chase. Every ball counts!")
else:
    st.warning("Tough chase. The pressure is on! ⚠️")

# --- Training Data Distribution (Wickets Fallen) ---
st.subheader("📊 Training Data Snapshot: Wickets Fallen Distribution")
plt.figure(figsize=(6, 3))
plt.hist(training_data["Innings Wickets"], bins=range(12), edgecolor="black", alpha=0.7)
plt.title("Distribution of Wickets Fallen in Training Data")
plt.xlabel("Wickets Fallen")
plt.ylabel("Frequency")
st.pyplot(plt)

# --- Footer ---
st.markdown(
    """
---
Made with ❤️ by a cricket fanatic & data scientist.
"""
)
