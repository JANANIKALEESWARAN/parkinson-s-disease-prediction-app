import streamlit as st
import numpy as np
import joblib

# Load the model
try:
    model = joblib.load("parkinsons.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("🧠 Parkinson's Disease Prediction")
st.write("Enter voice measurement values to predict Parkinson's Disease.")

# Input fields with precise steps and decimal format
Fo = st.number_input("MDVP:Fo(Hz)", 80.0, 300.0, 198.383, step=0.001, format="%.6f")
Fhi = st.number_input("MDVP:Fhi(Hz)", 100.0, 600.0, 215.203, step=0.001, format="%.6f")
Flo = st.number_input("MDVP:Flo(Hz)", 50.0, 300.0, 193.104, step=0.001, format="%.6f")
Jitter_percent = st.number_input("MDVP:Jitter(%)", 0.00001, 0.04360, 0.00212, step=0.00001, format="%.6f")
Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", 0.00001, 0.01, 0.00001, step=0.000001, format="%.6f")
RAP = st.number_input("MDVP:RAP", 0.0, 0.01, 0.00113, step=0.00001, format="%.6f")
PPQ = st.number_input("MDVP:PPQ", 0.0, 0.01, 0.00135, step=0.00001, format="%.6f")
DDP = st.number_input("Jitter:DDP", 0.0, 0.03, 0.00339, step=0.00001, format="%.6f")
Shimmer = st.number_input("MDVP:Shimmer", 0.0, 0.2, 0.01263, step=0.00001, format="%.6f")
Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", 0.0, 3.0, 0.111, step=0.001, format="%.6f")
APQ3 = st.number_input("Shimmer:APQ3", 0.0, 0.05, 0.0064, step=0.0001, format="%.6f")
APQ5 = st.number_input("Shimmer:APQ5", 0.0, 0.05, 0.00825, step=0.0001, format="%.6f")
APQ = st.number_input("MDVP:APQ", 0.0, 0.05, 0.00951, step=0.0001, format="%.6f")
DDA = st.number_input("Shimmer:DDA", 0.0, 0.05, 0.01919, step=0.0001, format="%.6f")
NHR = st.number_input("NHR", 0.0, 1.0, 0.00119, step=0.00001, format="%.6f")
HNR = st.number_input("HNR", 0.0, 50.0, 30.775, step=0.001, format="%.6f")
RPDE = st.number_input("RPDE", 0.0, 1.0, 0.465946, step=0.0001, format="%.6f")
DFA = st.number_input("DFA", 0.0, 1.0, 0.738703, step=0.0001, format="%.6f")
spread1 = st.number_input("spread1", -10.0, 0.0, -7.067931, step=0.0001, format="%.6f")
spread2 = st.number_input("spread2", 0.0, 1.0, 0.175181, step=0.0001, format="%.6f")
D2 = st.number_input("D2", 0.0, 5.0, 1.512275, step=0.0001, format="%.6f")
PPE = st.number_input("PPE", 0.0, 1.0, 0.09632, step=0.0001, format="%.6f")
scaler = joblib.load("scalar.pkl")
# Predict button
if st.button("Predict"):
    input_data = np.array([[Fo, Fhi, Flo, Jitter_percent, Jitter_Abs,
                            RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                            APQ3, APQ5, APQ, DDA, NHR, HNR,
                            RPDE, DFA, spread1, spread2, D2, PPE]])
    try:
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]
        result = "🧠 Parkinson's Detected" if prediction == 1 else "✅ No Parkinson's Detected"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
