# app.py
import streamlit as st
import numpy as np
import pickle

# Load model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("❤️ Heart Disease Prediction")

st.write("### Please fill in patient information:")

# Input form
age = st.number_input("Age", 1, 120, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    try:
        # Prepare input
        input_dict = {
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope
        }

        input_df = [input_dict]
        import pandas as pd
        input_df = pd.DataFrame(input_df)

        # Prediction
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        # Output
        if prediction == 1:
            st.error(f"⚠️ High risk of heart disease. Probability: {prob:.2f}")
        else:
            st.success(f"✅ Low risk of heart disease. Probability: {prob:.2f}")
    except Exception as e:
        st.exception(f"Prediction failed: {e}")
