import streamlit as st
import numpy as np
import xgboost as xgb
import os

# =========================
# Page settings
# =========================
st.set_page_config(
    page_title="Bladder Cancer Bone Metastasis Risk Calculator",
    layout="centered"
)

st.title("Bladder Cancer Bone Metastasis Risk Calculator")

st.write(
    "This web-based calculator estimates the probability of bone metastasis "
    "in patients with bladder cancer using an XGBoost model."
)

# =========================
# Sidebar inputs
# =========================
st.sidebar.header("Patient Variables")

# Age
Age = st.sidebar.selectbox(
    "Age",
    ["<60", "60–80", ">80"]
)

Age_map = {
    "<60": 0,
    "60–80": 1,
    ">80": 2
}

# Radiotherapy
Radiotherapy = st.sidebar.selectbox(
    "Radiotherapy",
    ["Yes", "No"]
)

Radiotherapy_map = {
    "Yes": 1,
    "No": 0
}

# Surgery
Surgery = st.sidebar.selectbox(
    "Surgery",
    ["No", "Yes"]
)

Surgery_map = {
    "No": 0,
    "Yes": 1
}

# Tumor size
Tumor_size = st.sidebar.selectbox(
    "Tumor size",
    ["≤4 cm", ">4 cm"]
)

Tumor_size_map = {
    "≤4 cm": 0,
    ">4 cm": 1
}

# T stage
T_stage = st.sidebar.selectbox(
    "T stage",
    ["T1", "T2", "T3", "T4"]
)

T_stage_map = {
    "T1": 0,
    "T2": 1,
    "T3": 2,
    "T4": 3
}

# N stage
N_stage = st.sidebar.selectbox(
    "N stage",
    ["N0", "N1", "N2", "N3"]
)

N_stage_map = {
    "N0": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3
}

# =========================
# Construct model input
# =========================
x = np.array([
    Age_map[Age],
    Radiotherapy_map[Radiotherapy],
    Surgery_map[Surgery],
    Tumor_size_map[Tumor_size],
    T_stage_map[T_stage],
    N_stage_map[N_stage]
]).reshape(1, 6)

# =========================
# Load model
# =========================
MODEL_FILE = "modelbladder.json"

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    return model

# =========================
# Prediction
# =========================
if st.button("Predict"):

    if not os.path.exists(MODEL_FILE):
        st.error("Model file 'modelbladder.json' was not found.")

    else:
        try:
            modelXGB = load_model()

            y_pred = modelXGB.predict_proba(x)

            probability = float(y_pred[0, 1]) * 100

            st.subheader("Prediction Result")

            st.metric(
                label="Probability of Bone Metastasis",
                value=f"{probability:.2f}%"
            )

            if probability >= 50:
                st.warning(
                    "The predicted probability of bone metastasis is relatively high."
                )
            else:
                st.success(
                    "The predicted probability of bone metastasis is relatively low."
                )

        except Exception as e:
            st.error("An error occurred while loading the model or making the prediction.")
            st.exception(e)

# =========================
# Disclaimer
# =========================
st.markdown("---")

st.caption(
    "Disclaimer: This calculator is intended for research purposes only "
    "and should not replace clinical judgment."
)
