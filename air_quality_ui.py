import streamlit as st
import numpy as np
import joblib

# Load model dan encoder
model = joblib.load('best_rf_model.pkl')
encoder = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="Prediksi Kualitas Udara", layout="wide")

# Custom CSS (fix background + text color + card style)
st.markdown("""
    <style>
        .stApp {
            background-image: linear-gradient(to bottom right, #e0f7fa, #ffffff);
            background-attachment: fixed;
            color: black !important;
        }
        .card-style {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        }
        h1, h3, p, label, .stTextInput > label, .stNumberInput > label {
            color: #004d40 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🌍 Prediksi Kualitas Udara</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan parameter lingkungan dan lihat prediksi kualitas udara secara langsung.</p>", unsafe_allow_html=True)

# Container 1 Halaman, dua kolom: input kiri, output kanan
container = st.container()
col1, col2 = container.columns([2, 1])

# Kiri - Input (dibungkus dalam card-style)
with col1:
    with st.container():
        st.markdown("<div class='card-style'>", unsafe_allow_html=True)
        st.markdown("### 🔧 Input Parameter")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            temperature = st.number_input("🌡️ Suhu (°C)", 0.0, 50.0, 25.0)
            pm25 = st.number_input("💨 PM2.5", 0.0, 500.0, 10.0)
            co = st.number_input("🟤 CO", 0.0, 10.0, 1.0)
        with col_b:
            humidity = st.number_input("💧 Kelembaban (%)", 0.0, 100.0, 60.0)
            pm10 = st.number_input("🌁 PM10", 0.0, 500.0, 20.0)
            proximity = st.number_input("🏭 Jarak Industri (km)", 0.0, 20.0, 5.0)
        with col_c:
            no2 = st.number_input("🧪 NO2", 0.0, 200.0, 15.0)
            so2 = st.number_input("🧪 SO2", 0.0, 200.0, 10.0)
            pop_density = st.number_input("👥 Kepadatan Penduduk", 0.0, 1000.0, 300.0)
        st.markdown("</div>", unsafe_allow_html=True)

# Kanan - Output Prediksi
with col2:
    with st.container():
        st.markdown("<div class='card-style'>", unsafe_allow_html=True)
        st.markdown("### 📊 Hasil Prediksi")

        if st.button("🔍 Prediksi Sekarang"):
            features = np.array([[temperature, humidity, pm25, pm10, no2, so2, co, proximity, pop_density]])
            pred = model.predict(features)[0]
            air_quality = encoder.inverse_transform([pred])[0]

            if air_quality == 'Good':
                st.success("✅ Udara Baik – Aman untuk beraktivitas 😊")
            elif air_quality == 'Moderate':
                st.warning("⚠️ Udara Sedang – Kurangi aktivitas luar 😐")
            elif air_quality == 'Hazardous':
                st.error("🚨 Udara Berbahaya – Hindari keluar rumah! 😷")
            else:
                st.info("ℹ️ Tidak diketahui – Perlu observasi lebih lanjut.")
        else:
            st.info("Masukkan data dan klik prediksi.")
        st.markdown("</div>", unsafe_allow_html=True)
