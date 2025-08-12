import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from features import feature_engineering

# =========================
# Load model and features
# =========================
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_cricket_model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_model_features():
    try:
        return joblib.load("model_features.pkl")
    except Exception as e:
        st.error(f"Error loading model features: {str(e)}")
        return None

model = load_model()
model_features = load_model_features()

if model is None or model_features is None:
    st.error("Failed to load model or features. Please check the files and try again.")
    st.stop()

# =========================
# Prediction + Calibration
# =========================
def predict_win_prob(df):
    """
    Predict win probability with minimal penalties for realistic ODI scenarios.
    Returns probability in % (1 to 99).
    """
    try:
        base_prob = model.predict_proba(df)[0][1] * 100  # Base ML output

        # --- Wickets penalty (extremely gentle) ---
        wickets_fallen = df["Innings Wickets"].iloc[0]
        wickets_remaining = 10 - wickets_fallen
        alpha = 0.5  # Minimal impact for wickets
        wicket_factor = (wickets_remaining / 10) ** alpha
        prob = base_prob * wicket_factor

        # --- RRR penalty (only for very high RRR) ---
        runs_remaining = df["Runs to Get"].iloc[0]
        balls_remaining = df["Balls Remaining"].iloc[0]
        rr_penalty = 1.0  # Default: no penalty
        if balls_remaining > 0:
            current_rrr = (runs_remaining / balls_remaining) * 6
            rrr_threshold = 9.0  # Skip penalty for RRR ~3.8
            if current_rrr > rrr_threshold:
                excess = current_rrr - rrr_threshold
                rr_penalty = 1 / (1 + np.exp(0.2 * excess - 1.0))  # Soft slope
                prob *= rr_penalty

        # --- Team strength boost (e.g., for Australia) ---
        prob *= 1.2  # Boost for strong teams

        # Clamp to realistic limits
        prob = max(min(prob, 99), 1)

        # Debug output
        st.write(f"Debug: Base Prob = {base_prob:.2f}%, "
                 f"Wicket Factor = {wicket_factor:.2f}, "
                 f"RRR Penalty = {rr_penalty:.2f}, "
                 f"Final Prob = {prob:.2f}%")
        return prob

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 50.0  # Fallback to neutral probability

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="🏏 Cricket Chase Win Predictor", layout="wide")
st.title("🏏 Cricket Chase Win Probability Predictor")
st.markdown("""
Predict the probability of winning the chase based on current match conditions.
Use the sidebar to input match details and explore what-if scenarios.
""")

# Sidebar inputs
st.sidebar.header("Match Inputs")
innings_runs = st.sidebar.number_input("Current Score", min_value=0, value=46, step=1)
innings_wickets = st.sidebar.slider("Wickets Fallen", min_value=0, max_value=10, value=2)
target_score = st.sidebar.number_input("Target Score", min_value=1, value=219, step=1)

runs_remaining_default = max(target_score - innings_runs, 0)
runs_remaining = st.sidebar.number_input(
    "Runs Remaining", min_value=0, value=runs_remaining_default, step=1
)
balls_remaining = st.sidebar.number_input(
    "Balls Remaining", min_value=0, max_value=300, value=273, step=1
)

# Validation
error_flag = False
if innings_runs > target_score:
    st.sidebar.error("Current Score cannot be greater than Target Score.")
    error_flag = True
if balls_remaining > 300:
    st.sidebar.error("Balls Remaining cannot exceed 300 (50 overs).")
    error_flag = True
if runs_remaining != runs_remaining_default:
    st.sidebar.warning(
        f"Runs Remaining ({runs_remaining}) != Target - Current Score ({runs_remaining_default})."
    )
if error_flag:
    st.stop()

# Prepare input data
input_df = pd.DataFrame({
    "Innings Runs": [innings_runs],
    "Innings Wickets": [innings_wickets],
    "Target Score": [target_score],
    "Runs to Get": [runs_remaining],
    "Balls Remaining": [balls_remaining],
})
input_df = input_df.reindex(columns=model_features)
try:
    input_df_fe = feature_engineering(input_df)
except Exception as e:
    st.error(f"Feature engineering error: {str(e)}")
    st.stop()
rrr_value = round((runs_remaining / balls_remaining * 6) if balls_remaining > 0 else 0, 2)

# Prediction
win_prob = predict_win_prob(input_df_fe)
lose_prob = 100 - win_prob

# =========================
# Display results
# =========================
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
    st.markdown(f"""
- **Current Score:** {innings_runs} 🏏  
- **Wickets Fallen:** {innings_wickets} ⚠️  
- **Target Score:** {target_score} 🎯  
- **Runs Remaining:** {runs_remaining} 🔥  
- **Balls Remaining:** {balls_remaining} ⏱️  
- **Required Run Rate:** {rrr_value} runs/over  
""")
    st.markdown("---")

# =========================
# What-if simulation
# =========================
st.subheader("📈 Explore What-If Scenarios")
scenario_variable = st.selectbox(
    "Select variable to simulate changes:",
    options=["Balls Remaining", "Wickets Fallen", "Runs Remaining"],
)

# Dynamic slider range
if scenario_variable == "Balls Remaining":
    var_min, var_max = 0, min(300, balls_remaining + 60)
elif scenario_variable == "Wickets Fallen":
    var_min, var_max = 0, 10
else:
    var_min, var_max = 0, max(runs_remaining, target_score - innings_runs)

var_values = st.slider(
    f"Adjust {scenario_variable} range",
    min_value=var_min,
    max_value=var_max,
    value=(var_min, var_max)
)

# Run simulation
try:
    x_vals = np.linspace(var_values[0], var_values[1], 50)
    probs = []
    for val in x_vals:
        sim_input = input_df.copy()
        sim_input.loc[0, scenario_variable] = val
        sim_input = sim_input.reindex(columns=model_features)
        sim_input_fe = feature_engineering(sim_input)
        sim_prob = predict_win_prob(sim_input_fe)
        probs.append(sim_prob)

    # Plot simulation
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(x_vals, probs, color="#0077FF", lw=3)
    ax2.set_title(f"Win Probability vs {scenario_variable}")
    ax2.set_xlabel(scenario_variable)
    ax2.set_ylabel("Win Probability (%)")
    ax2.grid(True)
    st.pyplot(fig2)
except Exception as e:
    st.error(f"Simulation error: {str(e)}")

# =========================
# Insights
# =========================
st.subheader("💡 Quick Insights")
if win_prob > 95:
    st.success("🏆 Dominating position! Almost certain win.")
elif win_prob > 60:
    st.success("Strong position! Keep the momentum 🏏🔥")
elif win_prob > 25:
    st.info("Competitive chase. Every ball counts!")
elif win_prob > 5:
    st.warning("Tough chase, but miracles happen! ⚠️")
else:
    st.error("Very slim chance, but cricket loves a comeback! 💪")

st.markdown("---")
st.markdown("Made with ❤️ by a cricket fanatic & data scientist.")
