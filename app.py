import streamlit as st
import librosa
import tsfel
import pandas as pd
import numpy as np
import joblib
import pickle
import io
import warnings

# Menggunakan StandardScaler dan SimpleImputer dari Scikit-Learn
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

# --- KONFIGURASI HALAMAN WEB ---
st.set_page_config(
    page_title="Smart Voice Access",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 1. KONFIGURASI NAMA FILE (WAJIB ADA 8 FILE)
# ==========================================
# A. Model Perintah (Buka/Tutup)
CMD_MODEL = 'random_forest_model.joblib'
CMD_ENCODER = 'label_encoder.joblib'
CMD_SCALER = 'scaler.joblib'
CMD_IMPUTER = 'imputer.joblib'
CMD_FEATURES = 'feature_columns.joblib'

# B. Model Security (Dio/Maulana/Anonymous)
SEC_MODEL = 'model_security_suara.pkl'
SEC_FEATURES = 'fitur_terpilih.pkl'
SEC_SCALER = 'scaler_security.joblib'

# ==========================================
# 2. LOAD SEMUA RESOURCE
# ==========================================
@st.cache_resource
def load_resources():
    try:
        # Load Resources Perintah
        cmd_model = joblib.load(CMD_MODEL)
        cmd_le = joblib.load(CMD_ENCODER)
        cmd_scaler = joblib.load(CMD_SCALER)
        cmd_imputer = joblib.load(CMD_IMPUTER)
        cmd_top_features = joblib.load(CMD_FEATURES)
        
        # Load Resources Security
        with open(SEC_MODEL, 'rb') as f:
            sec_model = pickle.load(f)
        with open(SEC_FEATURES, 'rb') as f:
            sec_features_list = pickle.load(f)
        
        # Load Scaler Security
        sec_scaler = joblib.load(SEC_SCALER)

        # Config TSFEL
        cfg_tsfel = tsfel.get_features_by_domain()

        return (cmd_model, cmd_le, cmd_scaler, cmd_imputer, cmd_top_features, 
                sec_model, sec_features_list, sec_scaler, cfg_tsfel)

    except FileNotFoundError as e:
        st.error(f"⚠️ File hilang: {e}. Pastikan 8 file (.joblib & .pkl) ada di folder.")
        return None
    except Exception as e:
        st.error(f"⚠️ Gagal memuat resource: {e}")
        return None

# ==========================================
# 3. EKSTRAKSI FITUR: SECURITY (MFCC++)
# ==========================================
def extract_security_features(audio_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        # 1. MFCC (40)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # 2. Delta (40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)
        
        # 3. Delta-Delta (40)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_delta2_mean = np.mean(mfcc_delta2.T, axis=0)
        
        # 4. ZCR (1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
        
        # 5. RMSE (1)
        rmse = np.mean(librosa.feature.rms(y=y).T, axis=0)
        
        # Gabung 122 Fitur
        features = np.hstack([mfcc_mean, mfcc_delta_mean, mfcc_delta2_mean, zcr, rmse])
        
        # Beri nama kolom 0-121
        col_names = list(range(len(features))) 
        
        return pd.DataFrame([features], columns=col_names)
        
    except Exception as e:
        st.error(f"Error Security Feature: {e}")
        return None

# ==========================================
# 4. EKSTRAKSI FITUR: COMMAND (TSFEL)
# ==========================================
def extract_command_features(audio_bytes, cfg_tsfel, fs=48000, n_mfcc=20):
    try:
        signal, sr = librosa.load(io.BytesIO(audio_bytes), sr=fs, mono=True)
        if len(signal) < fs * 0.1: return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsfel_df = tsfel.time_series_features_extractor(cfg_tsfel, signal, fs=sr, window_size=len(signal))

        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        mfcc_data = {}
        for i in range(n_mfcc):
            mfcc_data[f'MFCC_Mean_{i+1}'] = [mfccs_mean[i]]
            mfcc_data[f'MFCC_Std_{i+1}'] = [mfccs_std[i]]

        mfcc_df = pd.DataFrame(mfcc_data)
        tsfel_df.reset_index(drop=True, inplace=True)
        mfcc_df.reset_index(drop=True, inplace=True)
        return pd.concat([tsfel_df, mfcc_df], axis=1)

    except Exception as e:
        st.error(f"Error Command Feature: {e}")
        return None

# ==========================================
# 5. PIPELINE UTAMA (DENGAN THRESHOLD KETAT)
# ==========================================
def process_full_pipeline(audio_bytes, resources):
    (cmd_model, cmd_le, cmd_scaler, cmd_imputer, cmd_top_features, 
     sec_model, sec_features_list, sec_scaler, cfg_tsfel) = resources

    st.audio(audio_bytes, format='audio/wav')
    detected_identity = "Unknown" 

    # --- KONFIGURASI KEAMANAN ---
    CONFIDENCE_THRESHOLD = 80.0  # <--- UBAH INI: Minimal 80% mirip baru diterima
    
    # --- TAHAP 1: CEK IDENTITAS (SECURITY) ---
    with st.status("🕵️ Memverifikasi Identitas Suara...", expanded=True) as status:
        sec_raw_df = extract_security_features(audio_bytes)
        
        if sec_raw_df is not None:
            try:
                # 1. SCALING
                sec_input_scaled_full = sec_scaler.transform(sec_raw_df)
                
                # 2. FILTER FITUR
                target_features = [int(f) for f in sec_features_list]
                sec_input_final = sec_input_scaled_full[:, target_features]
                
                # 3. PREDIKSI LABEL & PROBABILITAS
                identity_pred = sec_model.predict(sec_input_final)[0]
                
                # Ambil probabilitas tertinggi
                probs = sec_model.predict_proba(sec_input_final)
                confidence_id = np.max(probs) * 100
                
                # Tampilkan hasil sementara
                st.write(f"Analisis Awal: **{identity_pred}**")
                st.write(f"Tingkat Keyakinan: **{confidence_id:.2f}%** (Syarat: >{CONFIDENCE_THRESHOLD}%)")
                
                # === LOGIKA PENOLAKAN KETAT (STRICT MODE) ===
                
                # Skenario 1: Terdeteksi sebagai Anonymous
                if identity_pred == "Anonymous":
                    status.update(label="❌ Akses Ditolak!", state="error")
                    st.error(f"⛔ **SUARA TIDAK DIKENALI**")
                    return 
                
                # Skenario 2: Terdeteksi Dio/Maulana TAPI Keyakinan Rendah (Teman Anda kena di sini)
                if confidence_id < CONFIDENCE_THRESHOLD:
                    status.update(label="⚠️ Akses Ditolak (Kurang Yakin)!", state="error")
                    st.error(f"⛔ **MIRIP {identity_pred.upper()}, TAPI BELUM CUKUP YAKIN**")
                    st.warning(f"Sistem mendeteksi kemiripan, namun tingkat keyakinan hanya {confidence_id:.2f}%. Demi keamanan, akses ditolak.")
                    return

                # Jika Lolos Semua Skenario
                detected_identity = identity_pred
                status.update(label=f"✅ Identitas Terverifikasi: {detected_identity}", state="complete")
                
            except Exception as e:
                st.error(f"Error Security Process: {e}")
                return
        else:
            st.error("Gagal membaca audio.")
            return

    # --- TAHAP 2: CEK PERINTAH (COMMAND) ---
    st.divider()
    with st.spinner(f"👋 Hai **{detected_identity}**, sedang menganalisis perintah Anda..."):
        
        cmd_raw_df = extract_command_features(audio_bytes, cfg_tsfel)
        
        if cmd_raw_df is not None:
            try:
                selected_feats = cmd_raw_df[cmd_top_features]
                imputed = cmd_imputer.transform(selected_feats)
                scaled = cmd_scaler.transform(imputed)
                
                cmd_pred_idx = cmd_model.predict(scaled)[0]
                cmd_label_full = cmd_le.inverse_transform([cmd_pred_idx])[0] 
                
                command = "Buka" if "buka" in cmd_label_full.lower() else "Tutup"
                
                if command == "Buka":
                    st.success(f"🔓 **BUKA** - Selamat Datang, {detected_identity}!", icon="🔓")
                    st.image("https://cdn-icons-png.flaticon.com/512/2997/2997573.png", width=150)
                else:
                    st.warning(f"🔒 **TUTUP** - Terima Kasih, {detected_identity}!", icon="🔒")
                    st.image("https://cdn-icons-png.flaticon.com/512/2997/2997645.png", width=150)

            except Exception as e:
                st.error(f"Error Command Process: {e}")

# ==========================================
# 6. UI
# ==========================================
st.title("🎙️ Smart Voice Access Control")
st.markdown("Verifikasi Biometrik (Dio/Maulana) + Perintah Suara (Buka/Tutup)")

res = load_resources()

if res:
    tab1, tab2 = st.tabs(["📁 Upload File", "🎤 Rekam Langsung"])
    with tab1:
        uploaded_file = st.file_uploader("Upload .WAV", type=['wav','mp3'])
        if uploaded_file: process_full_pipeline(uploaded_file.getvalue(), res)
    with tab2:
        audio_input = st.audio_input("Merekam")
        if audio_input: process_full_pipeline(audio_input.getvalue(), res)
else:
    st.error("❌ Sistem Gagal. Cek kelengkapan file.")
