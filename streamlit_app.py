import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ========== Load Model ==========
svm_tanpa = joblib.load("models/svm_tanpa_optimasi.pkl")
svm_ga = joblib.load("models/svm_dengan_ga.pkl")
svm_pso = joblib.load("models/svm_dengan_pso.pkl")

# ========== Evaluasi Model ==========
evaluasi_model = {
    "SVM Tanpa Optimasi": {
        "Train Accuracy": 90.75,
        "Test Accuracy": 80.5,
        "Precision": 0.8079,
        "Recall": 0.8050,
        "F1-Score": 0.8053,
        "Execution Time (s)": 0.41,
        "Best Params": "-"
    },
    "SVM + GA": {
        "Train Accuracy": 98.625,
        "Test Accuracy": 84.0,
        "Precision": 0.8428,
        "Recall": 0.8400,
        "F1-Score": 0.8400,
        "Execution Time (s)": 84.87,
        "Best Params": "C=5.2894, gamma=0.4545"
    },
    "SVM + PSO": {
        "Train Accuracy": 94.5,
        "Test Accuracy": 84.5,
        "Precision": 0.8445,
        "Recall": 0.8450,
        "F1-Score": 0.8434,
        "Execution Time (s)": 296.15,
        "Best Params": "C=51.7000, gamma=0.0375"
    }
}


# ========== Daftar Fitur yang Dipakai Saat Training ==========
feature_columns = [
    "Memiliki tanggung jawab dalam memastikan seluruh siswa memahami materi yang diajarkan.",
    "Merespon setiap pertanyaan dari murid",
    "Bersikap adil dan tidak membeda-bedakan siswa dalam proses pembelajaran.",
    "Memberikan penilaian dan umpan balik terhadap hasil belajar peserta didik secara jelas dan objektif.",
    "Bertindak sesuai dengan norma agama, hukum, sosial, dan budaya dalam mengajar.",
    "Menyelenggarakan kegiatan belajar mengajar sesuai kurikulum Merdeka",
    "Sikap dan kepribadian guru memberikan pengaruh positif terhadap motivasi belajar saya.",
    "Menjelaskan materi/jawaban pertanyaan dengan jelas dan memperkuat pemahaman",
    "Soal ujian/ulangan sangat relevan dengan capaian pembelajaran",
    "Berupaya mengembangkan potensi yang dimiliki setiap peserta didik.",
    "Mudah dihubungi saat jam kerja",
    "Membangun diskusi dengan nyaman",
    "Memberikan wawasan baru tentang masa sekolah SMP dan yang lebih tinggi",
    "Memahami kesulitan murid dalam proses belajar",
    "Memahami karakteristik setiap peserta didik dalam proses pembelajaran",
    "Memfasilitasi untuk kegiatan berdiskusi/belajar kelompok di kelas",
    "Metode pembelajaran yang digunakan oleh guru membantu saya memahami materi dengan baik.",
    "Selalu hadir tepat waktu dan disiplin dalam menjalankan tugas mengajar di kelas.",
    "Mata pelajaran yang diajarkan oleh guru tersebut",
    "Berpenampilan rapih"
]

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
        df.drop(columns=[target_col], inplace=True)

    nama_guru = df["Nama Lengkap"].values[0] if "Nama Lengkap" in df.columns else "Tidak diketahui"
    df.drop(columns=["Nama Lengkap"], errors='ignore', inplace=True)

    return df, nama_guru

# ========== UI ==========
st.set_page_config(page_title="Klasifikasi Guru Mumpuni", layout="wide")
st.title("📊 Aplikasi Klasifikasi Guru Mumpuni")
st.write("Upload file data guru untuk diproses oleh 3 model: SVM, SVM + GA, SVM + PSO")
st.subheader("📊 Evaluasi Model (Train & Test)")

df_eval = pd.DataFrame(evaluasi_model).T  # Transpose agar model di index
df_eval_display = df_eval.copy()
df_eval_display[["Train Accuracy", "Test Accuracy"]] = df_eval_display[["Train Accuracy", "Test Accuracy"]].round(2)
df_eval_display[["Precision", "Recall", "F1-Score"]] = df_eval_display[["Precision", "Recall", "F1-Score"]].round(4)

st.dataframe(df_eval_display)

# Visualisasi Train vs Test Accuracy
st.write("### 🔍 Perbandingan Akurasi (Train vs Test)")
df_acc = df_eval[["Train Accuracy", "Test Accuracy"]]
df_acc.plot(kind='bar')
plt.ylabel("Akurasi (%)")
plt.ylim(0, 100)
st.pyplot()


uploaded_file = st.file_uploader("📎 Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    if st.button("🚀 Proses Data"):
        try:
            df_input = pd.read_excel(uploaded_file)
            X, nama_guru = preprocess_data(df_input)

            # Sesuaikan kolom input agar cocok dengan fitur model
            X = X.reindex(columns=feature_columns, fill_value=0)

            # Model tanpa probabilitas → fallback ke .predict()
            try:
                pred_tanpa_score = svm_tanpa.predict_proba(X)[0][1] * 100
            except AttributeError:
                pred_class = svm_tanpa.predict(X)[0]
                pred_tanpa_score = 100.0 if pred_class == 1 else 0.0

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

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")
