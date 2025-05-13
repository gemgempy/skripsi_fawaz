import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# ========== Load Model ==========
svm_tanpa = joblib.load("models/svm_tanpa_optimasi.pkl")
svm_ga = joblib.load("models/svm_dengan_ga.pkl")
svm_pso = joblib.load("models/svm_dengan_pso.pkl")

# ========== Helper ==========
def preprocess_data(df):
    drop_columns = [
        "Timestamp", "Email Address", "Saat ini, Anda kelas berapa?",
        "Usia saat ini", "Mohon memilih satu orang guru yang saat ini mengajar anda",
        "Dengan ini, saya menyatakan bersedia berpartisipasi menjadi responden dalam penelitian ini. Jawaban yang saya berikan adalah jawaban yang sebenar-benarnya terjadi tanpa ada pengaruh dari hal/pihak manapun. Saya memahami bahwa seluruh informasi yang saya berikan dijaga kerahasiannya.",
        "Nomor Telepon untuk Giveaway 💰🤑", "Distribusi guru dan kelas",
        "Menguasai materi pelajaran dengan baik sehingga mudah dipahami oleh peserta didik.  ",
        "Dengan ini, saya menyatakan bersedia berpartisipasi menjadi responden dalam penelitian ini. Jawaban yang saya berikan adalah jawaban yang sebenar-benarnya terjadi tanpa ada pengaruh dari hal/pihak manapun. Saya memahami bahwa seluruh informasi yang b"
    ]

    df.drop(columns=drop_columns, inplace=True, errors='ignore')

    target_col = "Menurut anda apakah guru ini mumpuni dalam mengajar"

    # Parsikan nilai likert
    def parse_likert(x):
        for i in range(1, 6):
            if str(i) in str(x):
                return i
        return np.nan

    for col in df.columns:
        if col != target_col and df[col].dtype == object:
            df[col] = df[col].apply(parse_likert)

    df.fillna(df.mean(numeric_only=True), inplace=True)

    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    nama_guru = df["Nama Lengkap"].values[0] if "Nama Lengkap" in df.columns else "Tidak diketahui"
    df = df.drop(columns=["Nama Lengkap"], errors='ignore')
    
    return df, nama_guru

# ========== UI ==========
st.set_page_config(page_title="Klasifikasi Guru Mumpuni", layout="wide")
st.title("📊 Aplikasi Klasifikasi Guru Mumpuni")
st.write("Upload file data guru untuk diproses oleh 3 model: SVM, SVM + GA, SVM + PSO")

uploaded_file = st.file_uploader("📎 Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    if st.button("🚀 Proses Data"):
        df_input = pd.read_excel(uploaded_file)
        X, nama_guru = preprocess_data(df_input)
        X = X.reindex(columns=svm_ga.feature_names_in_, fill_value=0)
        X.fillna(0, inplace=True)

        # Model tanpa probabilitas
        pred_tanpa_score = svm_tanpa.predict_proba(X)[0][1] * 100

        # Model dengan probabilitas
        pred_ga = svm_ga.predict_proba(X)[0][1] * 100
        pred_pso = svm_pso.predict_proba(X)[0][1] * 100

        hasil = {
            "Model": ["SVM Tanpa Optimasi", "SVM + GA", "SVM + PSO"],
            "Persentase Mumpuni (%)": [pred_tanpa_score, pred_ga, pred_pso],
            "Klasifikasi": [
                "Mumpuni" if pred_tanpa_score >= 50 else "Tidak Mumpuni",
                "Mumpuni" if pred_ga >= 50 else "Tidak Mumpuni",
                "Mumpuni" if pred_pso >= 50 else "Tidak Mumpuni"
            ]
        }

        st.subheader("📌 Hasil Prediksi")
        st.write(f"**Nama Guru**: {nama_guru}")
        st.dataframe(pd.DataFrame(hasil))

        # Plot chart
        fig, ax = plt.subplots()
        ax.bar(hasil["Model"], hasil["Persentase Mumpuni (%)"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Persentase Mumpuni (%)")
        ax.set_title("Perbandingan Prediksi Antar Model")
        st.pyplot(fig)

