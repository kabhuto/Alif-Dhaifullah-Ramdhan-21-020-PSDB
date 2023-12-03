import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('gridrandomforestzscore.pkl', 'rb') as file_model:
    model_rf = pickle.load(file_model)

# UI components
st.title("Prediksi Penyakit Stroke Menggunakan Model Random Forest")

# Collect user input
age = st.number_input("Umur:", min_value=0, max_value=100, value=30)
hypertension = st.radio("Apakah memiliki penyakit hypertension?", ["0", "1"])
heart_disease = st.radio("Apakah memiliki penyakit jantung?", ["0", "1"])
avg_glucose_level = st.slider("Rata-rata gula darah:", min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI:", min_value=10.0, max_value=50.0, value=25.0)

# Prepare input data
input_feature = {
    'age': age,
    'hypertension': float(hypertension),
    'heart_disease': float(heart_disease),
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
}

# Convert to DataFrame
input_data = pd.DataFrame([input_feature])
if st.button ("cek prediksi"):
# Standardize input data
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Make prediction
    prediction = model_rf.predict(input_data_scaled)

# Show prediction result
    if prediction[0] == 0:
        st.write("Hasil Prediksi: Tidak Terkena stroke")
    else:
        st.write("Hasil Prediksi: Terkena stroke")
