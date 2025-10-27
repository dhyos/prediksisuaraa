import mysql.connector
from mysql.connector import Error
import json

# --- KONEKSI DATABASE ---
def connect_db():
    try:
        conn = mysql.connector.connect(
            host=103.245.38.100,
            user='root',          # ganti sesuai user MySQL-mu
            password='pkmkelompok123',          # isi password MySQL-mu
            database='suara_ai'
        )
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"Koneksi gagal: {e}")
        return None


# --- SIMPAN DATA REKAMAN ---
def simpan_rekaman(nama_file, audio_bytes, prediksi, confidence, durasi, fs, fitur_df):
    conn = connect_db()
    if conn is None:
        print("❌ Tidak dapat tersambung ke database.")
        return False
    
    try:
        cursor = conn.cursor()
        fitur_json = json.dumps(fitur_df.to_dict()) if fitur_df is not None else None
        
        sql = """
        INSERT INTO rekaman_suara
        (nama_file, audio, prediksi, confidence, durasi, fs, fitur_jumlah, fitur_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(sql, (
            nama_file,
            audio_bytes,
            prediksi,
            float(confidence),
            float(durasi),
            int(fs),
            len(fitur_df.columns) if fitur_df is not None else 0,
            fitur_json
        ))
        conn.commit()
        print("✅ Data rekaman berhasil disimpan ke database.")
        return True
    
    except Error as e:
        print(f"Error saat menyimpan: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


# --- TAMPILKAN SEMUA DATA ---
def ambil_semua_rekaman():
    conn = connect_db()
    if conn is None:
        return []
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM rekaman_suara ORDER BY waktu_rekam DESC")
        hasil = cursor.fetchall()
        return hasil
    except Error as e:
        print(f"Error saat mengambil data: {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

