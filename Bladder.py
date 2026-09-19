import streamlit as st
import numpy as np
import pickle
import os

st.title(
    'Application of Machine Learning Methods to Predict '
    'Bone Metastases in Bladder Cancer Patients'
)

st.sidebar.subheader('Variables')

Age = st.sidebar.selectbox('Age', ['<60', '60-80',">80"])
Age_map = {'<60': 0, '60-80': 1,">80":2}

Radiation = st.sidebar.selectbox('Radiotherapy', ['No', 'Yes'])
Radiation_map = {'No': 0, 'Yes': 1}

Surgery = st.sidebar.selectbox('Surgery', ['No', 'Yes'])
Surgery_map = {'No': 0, 'Yes': 1}

Tumorsize = st.sidebar.selectbox('Tumor size', ['<4 cm', '≥4 cm'])
Tumorsize_map = {'<4 cm': 0, '≥4 cm': 1}

T_stage = st.sidebar.selectbox(
    'T stage',
    ['T1', 'T2', 'T3', 'T4']
)
T_stage_map = {
    'T1': 0,
    'T2': 1,
    'T3': 2,
    'T4': 3
}

N_stage = st.sidebar.selectbox(
    'N stage',
    ['N0', 'N1', 'N2', 'N3']
)
N_stage_map = {
    'N0': 0,
    'N1': 1,
    'N2': 2,
    'N3': 3
}

filename = 'modelbladder.txt'

x = np.array([
    Age_map[Age],
    Radiation_map[Radiation],
    Surgery_map[Surgery],
    Tumorsize_map[Tumorsize],
    T_stage_map[T_stage],
    N_stage_map[N_stage]
]).reshape(1, 6)

if st.button('Predict'):

    if os.path.exists(filename):

        with open(filename, 'rb') as f:
            modelXGB = pickle.load(f)

        y_pred = modelXGB.predict_proba(x)

        probability = y_pred[0, 1] * 100

        st.header(
            f'Probability of bone metastasis: {probability:.2f}%'
        )

    else:
        st.error('Model file not found.')
