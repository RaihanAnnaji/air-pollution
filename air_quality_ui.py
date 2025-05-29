import streamlit as st
import numpy as np
import joblib

# Load model dan encoder
model = joblib.load('best_rf_model.pkl')
encoder = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="Prediksi Kualitas Udara", layout="centered")

st.title("Prediksi Kualitas Udara 🌍")
st.markdown("Masukkan parameter lingkungan untuk memprediksi kualitas udara:")

# Input fitur dari pengguna
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
pm25 = st.number_input("PM2.5", 0.0, 500.0, 10.0)
pm10 = st.number_input("PM10", 0.0, 500.0, 20.0)
no2 = st.number_input("NO2", 0.0, 200.0, 15.0)
so2 = st.number_input("SO2", 0.0, 200.0, 10.0)
co = st.number_input("CO", 0.0, 10.0, 1.0)
proximity = st.number_input("Proximity to Industrial Areas (km)", 0.0, 20.0, 5.0)
pop_density = st.number_input("Population Density", 0.0, 1000.0, 300.0)

# Tombol prediksi
if st.button("Prediksi"):
    features = np.array([[temperature, humidity, pm25, pm10, no2, so2, co, proximity, pop_density]])
    pred = model.predict(features)[0]
    air_quality = encoder.inverse_transform([pred])[0]
    
    st.subheader(f"Hasil Prediksi: {air_quality}")
    
    if air_quality == 'Good':
        st.success("Kualitas udara bagus 😊")
    elif air_quality == 'Moderate':
        st.warning("Kualitas udara sedang 😐")
    elif air_quality == 'Hazardous':
        st.error("Kualitas udara berbahaya! 🚨")
    else:
        st.info("Perlu pengamatan lebih lanjut.")

