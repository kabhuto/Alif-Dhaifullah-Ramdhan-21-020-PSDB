import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Memuat model yang telah dilatih
with open('gridrandomforestzscore.pkl', 'rb') as file_model:
    model_rf = pickle.load(file_model)

# Komponen UI
st.title("Prediksi Penyakit Stroke Menggunakan Model Random Forest")

# Mengumpulkan input pengguna
umur = st.number_input("Umur:", min_value=0, max_value=100, value=30)
hipertensi = st.radio("Apakah memiliki penyakit hipertensi?", ["0", "1"])
penyakit_jantung = st.radio("Apakah memiliki penyakit jantung?", ["0", "1"])
pernah_menikah = st.radio("Apakah pernah menikah?", ["0", "1"])
rata_gula_darah = st.slider("Rata-rata gula darah:", min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI:", min_value=10.0, max_value=50.0, value=25.0)

# Tombol untuk memicu prediksi
if st.button("Prediksi"):
    # Menyiapkan data input
    fitur_input = {
        'umur': umur,
        'hipertensi': float(hipertensi),
        'penyakit_jantung': float(penyakit_jantung),
        'pernah_menikah': float(pernah_menikah),
        'rata_gula_darah': rata_gula_darah,
        'bmi': bmi,
    }

    # Mengonversi ke DataFrame
    data_input = pd.DataFrame([fitur_input])

    # Menstandarisasi data input
    scaler = StandardScaler()
    data_input_scaled = scaler.fit_transform(data_input)

    # Melakukan prediksi
    prediksi = model_rf.predict(data_input_scaled)

    # Menampilkan hasil prediksi
    if prediksi[0] == 0:
        st.write("Hasil Prediksi: Tidak Terkena stroke")
    else:
        st.write("Hasil Prediksi: Terkena stroke")
