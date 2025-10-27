import streamlit as st
import librosa
import tsfel
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

# 🔗 koneksi database (Pastikan file suara_db.py ada)
try:
    from suara_db import simpan_rekaman, ambil_semua_rekaman 
except ImportError:
    st.error("❌ Gagal mengimpor 'suara_db.py'. Pastikan file tersebut ada di folder yang sama.")
    def simpan_rekaman(**kwargs): pass
    def ambil_semua_rekaman(): return []

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Suara (Buka/Tutup)",
    page_icon="🎙️",
    layout="centered"
)

# --- LOAD MODEL DAN PRA-PEMROSES ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('model_suara_classifier.joblib')
        scaler = joblib.load('scaler.joblib')
        imputer = joblib.load('imputer.joblib')
        cfg_statistical = tsfel.get_features_by_domain("statistical")
        
        # Ambil daftar kolom yang diharapkan model
        dummy_signal = np.random.rand(48000)
        dummy_features = tsfel.time_series_features_extractor(
            cfg_statistical, dummy_signal, fs=48000, window_size=len(dummy_signal)
        )
        numeric_cols = list(dummy_features.columns)
        return model, scaler, imputer, cfg_statistical, numeric_cols
    except Exception as e:
        st.error(f"⚠️ Gagal memuat model atau preprocessor: {e}")
        return None, None, None, None, None

# --- EKSTRAKSI FITUR STATISTIK ---
def extract_statistical_features(audio_bytes, cfg_statistical, fs=48000):
    try:
        signal, sr = librosa.load(audio_bytes, sr=fs, mono=True)
        if len(signal) < fs*0.1:
            st.warning("Audio terlalu pendek untuk diproses.")
            return None, 0
        features_df = tsfel.time_series_features_extractor(cfg_statistical, signal, fs=sr, window_size=len(signal))
        return features_df, len(signal)/sr
    except Exception as e:
        st.error(f"❌ Gagal ekstraksi fitur: {e}")
        return None, 0

# --- MUAT SUMBER DAYA ---
model, scaler, imputer, cfg_statistical, expected_cols = load_resources()

# --- ANTARMUKA ---
st.title("🎙️ Prediksi Suara (Buka / Tutup / Unknown)")
st.write("Unggah file audio atau rekam langsung dari mikrofon.")

audio_input = st.audio_input("Rekam atau unggah suara Anda:")

if audio_input is not None and model is not None:
    audio_bytes = io.BytesIO(audio_input.read())
    st.audio(audio_bytes, format='audio/wav')
    
    with st.spinner("🔍 Mengekstrak fitur statistik..."):
        features_df, durasi = extract_statistical_features(audio_bytes, cfg_statistical)
    
    if features_df is not None:
        try:
            features_df = features_df[expected_cols]  # pastikan urutan kolom sesuai model
            features_imputed = imputer.transform(features_df)
            features_scaled = scaler.transform(features_imputed)
            
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)*100
            
            # Tampilkan hasil
            if prediction == 'buka':
                st.success(f"✅ Hasil: BUKA ({confidence:.1f}%)")
            elif prediction == 'tutup':
                st.error(f"🔒 Hasil: TUTUP ({confidence:.1f}%)")
            else:
                st.warning(f"🤔 Hasil: SUARA TIDAK DIKETAHUI ({confidence:.1f}%)")
            
            # Simpan ke database
            nama_file = "rekaman_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") + ".wav"
            simpan_rekaman(
                nama_file=nama_file,
                audio_bytes=audio_input.read(),
                prediksi=prediction,
                confidence=confidence,
                durasi=durasi,
                fs=48000,
                fitur_df=features_df
            )
            
            # Rincian probabilitas
            with st.expander("Lihat Rincian Probabilitas"):
                st.dataframe(pd.DataFrame([probabilities], columns=model.classes_))
            
        except KeyError as e:
            st.error(f"⚠️ Kolom fitur tidak cocok: {e}")
        except Exception as e:
            st.error(f"⚠️ Terjadi kesalahan saat prediksi: {e}")

# Tampilkan riwayat rekaman
st.markdown("---")
st.subheader("📜 Riwayat Prediksi Suara")
try:
    rekaman_data = ambil_semua_rekaman()
    if len(rekaman_data) > 0:
        df_rekaman = pd.DataFrame(rekaman_data)
        st.dataframe(df_rekaman[['id','nama_file','waktu_rekam','prediksi','confidence']])
    else:
        st.info("Belum ada rekaman tersimpan.")
except Exception as e:
    st.error(f"Gagal mengambil riwayat: {e}")
