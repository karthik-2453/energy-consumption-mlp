import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model
model = load_model('../models/model.h5', compile=False)
scaler = pickle.load(open('../models/scaler.pkl', 'rb'))

# Page config
st.set_page_config(page_title="⚡ Energy Predictor")

# 🎨 Vibrant Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a, #9333ea);
    color: white;
}
h1 {
    color: #facc15;
    text-align: center;
}
.stButton>button {
    background: linear-gradient(45deg, #22c55e, #4ade80);
    color: white;
    font-size: 18px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("⚡ Energy Consumption Predictor")

# 🔹 Inputs
col1, col2 = st.columns(2)

with col1:
    reactive_power = st.number_input("Global Reactive Power", value=0.1)
    voltage = st.number_input("Voltage", value=220.0)
    intensity = st.number_input("Global Intensity", value=10.0)

with col2:
    sub1 = st.number_input("Sub Metering 1", value=1.0)
    sub2 = st.number_input("Sub Metering 2", value=1.0)
    sub3 = st.number_input("Sub Metering 3", value=1.0)

hour = st.slider("Hour", 0, 23, 12)
day = st.slider("Day", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)

# Prediction function
def predict(data):
    arr = np.array(data).reshape(1, -1)
    arr = scaler.transform(arr)
    pred = model.predict(arr)
    return float(pred[0][0])

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# 🔹 Button
if st.button("🚀 Predict Energy"):
    features = [
        reactive_power,
        voltage,
        intensity,
        sub1,
        sub2,
        sub3,
        hour,
        day,
        month
    ]

    result = predict(features)

    st.success(f"⚡ Predicted Energy: {result:.3f} kW")

    # Save result
    st.session_state.history.append(result)

    # 🔔 Alerts
    if result > 5:
        st.error("🚨 High Energy Usage!")
    elif result > 2:
        st.warning("⚠️ Moderate Usage")
    else:
        st.info("✅ Efficient Usage")
        
    # 🔹 📊 Line Graph with Legend
if len(st.session_state.history) > 0:
    plt.figure()
    plt.plot(st.session_state.history, marker='o', label="Predicted Energy")
    plt.xlabel("Predictions")
    plt.ylabel("Energy Consumption")
    plt.title("Energy Trend")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

    # 💡 Recommendations
    st.write("### 💡 Recommendations")

    if result > 5:
        st.write("- Reduce usage of high-power appliances")
        st.write("- Avoid peak hour electricity usage (6PM–10PM)")
        st.write("- Check for faulty or energy-draining devices")
        st.write("- Use voltage stabilizers to prevent fluctuations")

    elif result > 2:
        st.write("- Use appliances efficiently")
        st.write("- Avoid running multiple heavy devices together")
        st.write("- Maintain stable voltage supply")
        st.write("- Switch to energy-efficient appliances")

    else:
        st.write("- Good energy management 👍")
        st.write("- Continue using efficient appliances")
        st.write("- Use LED lighting to save energy")
