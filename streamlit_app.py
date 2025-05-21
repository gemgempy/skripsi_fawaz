import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ========== Load Model ==========
svm_tanpa = joblib.load("models/svm_tanpa_optimasi.pkl")
svm_ga = joblib.load("models/svm_dengan_ga.pkl")
svm_pso = joblib.load("models/svm_dengan_pso.pkl")

# ========== Evaluasi Model ==========
evaluasi_model = {
    "SVM Tanpa Optimasi": {
        "Train Accuracy": 90.8,
        "Test Accuracy": 80.5,
        "Precision": 0.808,
        "Recall": 0.805,
        "F1-Score": 0.805,
        "Execution Time (s)": 0.11,
        "Best Params": "-"
    },
    "GA SVM": {
        "Train Accuracy": 98.5,
        "Test Accuracy": 84.0,
        "Precision": 0.843,
        "Recall": 0.84,
        "F1-Score": 0.84,
        "Execution Time (s)": 30.19,
        "Best Params": "C=5.2894, gamma=0.4545"
    },
    "PSO SVM": {
        "Train Accuracy": 94.8,
        "Test Accuracy": 84.5,
        "Precision": 0.844,
        "Recall": 0.845,
        "F1-Score": 0.843,
        "Execution Time (s)": 99.37,
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

    nama_guru = df["Nama Lengkap"].values[0] if "Nama Lengkap" in df.columns else "Tidak diketahui"

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

    df.drop(columns=["Nama Lengkap"], errors='ignore', inplace=True)

    return df, nama_guru


# ========== UI ==========
st.set_page_config(page_title="Klasifikasi Guru Mumpuni", layout="wide")
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 1rem;">
        <img src="http://smpn1tangerang.sch.id/wp-content/uploads/2020/03/logosmpn1tangerang.png" 
             alt="Logo SMPN 1 Tangerang" width="80" height="80">
        <h1 style="margin: 0;"> Aplikasi Klasifikasi Kualitas Guru SMP Negeri 1 Tangerang</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("Upload file data guru untuk diproses oleh 3 model: SVM, SVM + GA, SVM + PSO")
uploaded_file = st.file_uploader("📎 Upload file Excel (.xlsx)", type=["xlsx"])
process_clicked = st.button("🚀 Proses Data")

st.subheader("📊 Evaluasi Model (Train & Test)")

df_eval = pd.DataFrame(evaluasi_model).T  
df_eval_display = df_eval.copy()
df_eval_display[["Train Accuracy", "Test Accuracy"]] = df_eval_display[["Train Accuracy", "Test Accuracy"]].round(2)
df_eval_display[["Precision", "Recall", "F1-Score"]] = df_eval_display[["Precision", "Recall", "F1-Score"]].round(4)

st.dataframe(df_eval_display)

# Visualisasi Train vs Test Accuracy
with st.expander("🔍 Perbandingan Akurasi (Train vs Test)", expanded=True):
    df_acc = df_eval[["Train Accuracy", "Test Accuracy"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    df_acc.plot(kind='bar', ax=ax)
    ax.set_ylabel("Akurasi (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)


if uploaded_file and process_clicked:
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
            "Model": ["SVM Tanpa Optimasi", "GA SVM", "PSO SVM"],
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

        # Custom visualisasi dengan "tabung" berwarna gradasi dan label kualitas
        fig, ax = plt.subplots(figsize=(9, 7))
        
        label_kualitas = ["Sangat Tidak Mumpuni", "Tidak Mumpuni", "Kurang Mumpuni", "Mumpuni", "Sangat Mumpuni"]
        warna_gradasi = ["#cc0000", "#ff6600", "#ffcc00", "#66cc66", "#009933"]
        tinggi_segment = 20
        
        for idx, (model, score) in enumerate(zip(hasil["Model"], hasil["Persentase Mumpuni (%)"])):
            ax.add_patch(patches.Rectangle((idx * 1.8, 0), 1.2, 100, fill=False, edgecolor='black', linewidth=2))
        
            for i in range(5):
                bottom = i * tinggi_segment
                color = warna_gradasi[i]
                ax.add_patch(patches.Rectangle((idx * 1.8, bottom), 1.2, tinggi_segment, color=color, alpha=0.8))
        
            pred_height = score
            ax.add_patch(patches.Rectangle((idx * 1.8, 0), 1.2, pred_height, color="#ff9966", alpha=1, zorder=5))  
            ax.plot([idx * 1.8, (idx * 1.8) + 1.2], [pred_height, pred_height], color='black', linewidth=3, zorder=6)
        
            ax.text(idx * 1.8 + 0.6, pred_height - 5, f"{score:.1f}%", ha='center', va='center', fontsize=11, fontweight='bold', color='black', zorder=7)
        
        ax.set_yticks([10, 30, 50, 70, 90])
        ax.set_yticklabels(label_kualitas)
        ax.set_ylabel("Persentase Kemumpunian (%)")
        
        ax.set_xlim(-0.5, len(hasil["Model"]) * 1.8 - 0.6)
        ax.set_xticks([i * 1.8 + 0.6 for i in range(len(hasil["Model"]))])
        ax.set_xticklabels(hasil["Model"], fontsize=10)
        
        from matplotlib.lines import Line2D
        legend_elements = [patches.Patch(facecolor=color, edgecolor='black', label=label) 
                           for label, color in zip(label_kualitas, warna_gradasi)]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08), 
                  fancybox=True, shadow=False, ncol=3, title="Tingkat Kemumpunian")
        
        ax.set_title("Visualisasi Prediksi Kemumpunian Guru oleh Tiga Model", fontsize=13, weight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        st.pyplot(fig)
        

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")
