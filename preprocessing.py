import pandas as pd
import numpy as np

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
        if pd.isna(x):
            return np.nan
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
    df = df.drop(columns=["Nama Lengkap"], errors='ignore')

    return df, nama_guru
