import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('best_heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Heart Disease Prediction App')

st.write("### Enter patient details below:")

# --- Input UI ---

age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 140)
cholesterol = st.number_input("Cholesterol", 100, 600, 289)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate", 60, 220, 172)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, step=0.1, value=0.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# --- Manual One-Hot Encoding Mappings ---
def one_hot_encode(value, categories):
    return [1 if value == cat else 0 for cat in categories]

# All categories (same order used during training)
sex_encoded = one_hot_encode(sex, ["M", "F"])
cp_encoded = one_hot_encode(chest_pain, ["TA", "ATA", "NAP", "ASY"])
ecg_encoded = one_hot_encode(resting_ecg, ["Normal", "ST", "LVH"])
angina_encoded = one_hot_encode(exercise_angina, ["Y", "N"])
slope_encoded = one_hot_encode(st_slope, ["Up", "Flat", "Down"])

# --- Final Input Vector (20 features) ---
numerical = [age, resting_bp, cholesterol, fasting_bs, max_hr, oldpeak]

final_input = np.array([numerical + sex_encoded + cp_encoded +
                        ecg_encoded + angina_encoded + slope_encoded])

# --- Prediction ---
if st.button("Predict"):
    try:
        prediction = model.predict(final_input)
        if prediction[0] == 1:
            print("value:", prediction[0])
            st.error("⚠️ High risk of heart disease.")
        else:
            st.success("✅ Low risk of heart disease.")
    except Exception as e:
        st.exception(f"Model prediction failed: {e}")
