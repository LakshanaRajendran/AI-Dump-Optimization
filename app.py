import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from rl_optimizer import get_state
from utils import efficiency_score, estimate_energy
from rl_optimizer import get_best_action
# Load models
reg_model = joblib.load("reg_model.pkl")
clf_model = joblib.load("clf_model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="AI Dump System", layout="wide")

st.title("🚛 AI Dump Optimization System")

# Sidebar inputs
st.sidebar.header("⚙️ Input Parameters")

max_angle = st.sidebar.slider("Max Angle", 40, 60, 50)
hold_time = st.sidebar.slider("Hold Time", 1, 10, 4)
total_dump_time = st.sidebar.slider("Total Dump Time", 5, 15, 8)
vibration_rms = st.sidebar.slider("Vibration RMS", 0.5, 3.0, 1.5)
vibration_peak = st.sidebar.slider("Vibration Peak", 1.0, 3.5, 2.0)
initial_load = st.sidebar.slider("Initial Load", 800, 1500, 1000)

# Feature engineering
efficiency_ratio = max_angle / total_dump_time
vibration_effectiveness = vibration_peak / hold_time
dump_aggressiveness = vibration_rms * max_angle

input_data = np.array([[
    max_angle, hold_time, total_dump_time,
    vibration_rms, vibration_peak, initial_load,
    efficiency_ratio, vibration_effectiveness, dump_aggressiveness
]])

# Prediction
leftover = reg_model.predict(input_data)[0]
state = get_state(leftover)

action = get_best_action(leftover)
severity = le.inverse_transform(clf_model.predict(input_data))[0]

tree_preds = [tree.predict(input_data)[0] for tree in reg_model.estimators_]
confidence = 100 - np.std(tree_preds)

eff_score = efficiency_score(leftover)
energy = estimate_energy(max_angle, hold_time, vibration_rms)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analytics", "🧪 Simulation"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("📊 Live Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric("Leftover %", f"{leftover:.2f}%")
    col2.metric("Efficiency Score", f"{eff_score:.2f}")
    col3.metric("Confidence", f"{confidence:.2f}%")

    st.write(f"**Severity Level:** {severity}")
    st.write(f"**Estimated Energy Usage:** {energy:.2f}")

    # Gauge Chart (VERY COOL)
    fig = px.pie(
        names=["Dumped", "Leftover"],
        values=[100-leftover, leftover],
        title="Material Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("📈 Dump Behavior Analysis")

    # Generate angle vs leftover data
    angles = np.linspace(40, 60, 20)
    leftovers = []

    for a in angles:
        temp = np.array([[a, hold_time, total_dump_time,
                          vibration_rms, vibration_peak, initial_load,
                          a/total_dump_time, vibration_peak/hold_time, vibration_rms*a]])
        pred = reg_model.predict(temp)[0]
        leftovers.append(pred)

    df_graph = pd.DataFrame({
        "Angle": angles,
        "Predicted Leftover": leftovers
    })

    # Line graph
    fig1 = px.line(df_graph, x="Angle", y="Predicted Leftover",
                   title="Angle vs Leftover Projection",
                   markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    # Bar chart
    fig2 = px.bar(df_graph, x="Angle", y="Predicted Leftover",
                  title="Dump Efficiency Trend")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("🧪 Simulation Lab")

    col1, col2 = st.columns(2)

    with col1:
        angle_sim = st.slider("Sim Angle", 40, 60, 50, key="sim1")
        hold_sim = st.slider("Sim Hold", 1, 10, 4, key="sim2")

    with col2:
        vib_sim = st.slider("Sim Vibration", 0.5, 3.0, 1.5, key="sim3")
        time_sim = st.slider("Sim Time", 5, 15, 8, key="sim4")

    sim_input = np.array([[
        angle_sim, hold_sim, time_sim,
        vib_sim, vibration_peak, initial_load,
        angle_sim/time_sim, vibration_peak/hold_sim, vib_sim*angle_sim
    ]])

    sim_leftover = reg_model.predict(sim_input)[0]

    st.success(f"Predicted Leftover: {sim_leftover:.2f}%")

st.subheader("🤖 AI Optimization Suggestion")
if action == "increase_angle":
    msg = "Increase angle by 3°"
elif action == "increase_hold":
    msg = "Increase hold time by 2 sec"
else:
    msg = "Enable vibration assist"

st.success(msg)

# Comparison chart
compare_df = pd.DataFrame({
    "Scenario": ["Current", "Simulated"],
    "Leftover": [leftover, sim_leftover]
})

fig3 = px.bar(compare_df, x="Scenario", y="Leftover",
                title="Current vs Simulated Comparison")
st.plotly_chart(fig3, use_container_width=True)