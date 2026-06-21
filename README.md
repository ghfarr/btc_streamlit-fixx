# ₿ BTC Price Prediction Dashboard (Streamlit)

Dashboard prediksi harga **Bitcoin** menggunakan kombinasi data historis (CoinGecko) +
sentimen Twitter real (HuggingFace dataset) + 3 model machine learning
(**Linear Regression**, **Random Forest**, **LSTM**) untuk dua horizon: **t+1 hari** dan **t+30 hari**.

Built with [Streamlit](https://streamlit.io) · siap deploy ke Streamlit Community Cloud lewat GitHub.

---

## 🚀 Cara Deploy ke Streamlit Cloud (lewat GitHub)

### 1. Push folder ini ke GitHub

```bash
# Di komputer kamu, masuk ke folder ini lalu:
git init
git add .
git commit -m "Initial commit: BTC predictor streamlit app"

# Buat repo kosong di GitHub (lewat web), lalu:
git remote add origin https://github.com/<USERNAME-KAMU>/btc-predictor-streamlit.git
git branch -M main
git push -u origin main
```

> **PENTING:** Pastikan folder `data/` dengan semua file `.parquet` ikut ke-push.
> Itu bukan auto-generate — itu hasil precompute yang harus di-bundle ke repo.

### 2. Connect ke Streamlit Community Cloud

1. Buka https://share.streamlit.io
2. Login pakai akun **GitHub** kamu
3. Klik **"New app"** → **"From existing repo"**
4. Pilih repo `btc-predictor-streamlit`
5. Branch: `main`
6. Main file path: `app.py`
7. Klik **Deploy** → tunggu ~2 menit

URL public-nya akan jadi sesuatu seperti:
`https://<repo-kamu>.streamlit.app`

Selesai! Tiap kali ada `git push` baru ke `main`, app otomatis re-deploy. 🎉

---

## 🏃 Run Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

App akan jalan di http://localhost:8501

---

## 📁 Struktur Folder

```
btc_streamlit/
├── app.py                  ← main Streamlit app (yang di-deploy)
├── precompute.py           ← script training pipeline (BUKAN dipakai saat deploy)
├── requirements.txt        ← deps untuk Streamlit Cloud (ringan, no TF)
├── requirements-train.txt  ← deps tambahan kalau mau retrain (TF, datasets, dll.)
├── .streamlit/
│   └── config.toml         ← theme & server config
├── .gitignore
├── README.md               ← file ini
└── data/                   ← hasil precompute (HARUS ikut di-push ke GitHub!)
    ├── btc.parquet                ← harga BTC harian 2021–2025
    ├── sentiment.parquet          ← daily aggregate VADER score
    ├── tweets_sample.parquet      ← sampling 5k tweet untuk browsing
    ├── features.parquet           ← features siap-train (lag/MA/std/return + sentimen)
    ├── feature_cols.json          ← list nama fitur
    ├── predictions.parquet        ← prediksi LR/RF/LSTM × val/test × t+1/t+30
    ├── metrics.parquet            ← RMSE/MAE/MAPE per model × split × horizon
    ├── metrics.csv                ← versi CSV-nya (mudah dibaca manual)
    └── btc_2021_01_01_to_2025_12_31.csv   ← raw CSV (jaga-jaga / reproducibility)
```

---

## 🔄 Re-train Pipeline (Opsional)

Kalau mau re-train (misal: ganti hyperparameter, tambah data, dll.):

1. Buka Google Colab yang sama dengan notebook training
2. Pastikan semua variabel seperti `train`, `val`, `test`, `all_preds`, `results_df`, `feature_cols`, `tweets_df` masih ada di memori
3. Jalankan cell berikut di Colab:

\```python
import json, shutil
from pathlib import Path

DATA_DIR = Path('/content/data')
DATA_DIR.mkdir(exist_ok=True)

btc.to_parquet(DATA_DIR / 'btc.parquet', index=False)
sentiment.to_parquet(DATA_DIR / 'sentiment.parquet', index=False)
df.to_parquet(DATA_DIR / 'features.parquet', index=False)

with open(DATA_DIR / 'feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)

preds_long_rows = []
for horizon_label in ["t+1", "t+30"]:
    d_va = all_preds[horizon_label]["dates_val"]
    d_te = all_preds[horizon_label]["dates_test"]
    y_va = all_preds[horizon_label]["y_val"]
    y_te = all_preds[horizon_label]["y_test"]
    for model_name in ["LinearRegression", "RandomForest", "LSTM"]:
        pv = all_preds[horizon_label]["preds_val"][model_name]
        pt = all_preds[horizon_label]["preds_test"][model_name]
        for d, t, p in zip(d_va, y_va, pv):
            preds_long_rows.append({"horizon": horizon_label, "model": model_name,
                                    "split": "validation", "date": pd.Timestamp(d),
                                    "y_true": float(t), "y_pred": float(p)})
        for d, t, p in zip(d_te, y_te, pt):
            preds_long_rows.append({"horizon": horizon_label, "model": model_name,
                                    "split": "test", "date": pd.Timestamp(d),
                                    "y_true": float(t), "y_pred": float(p)})

pd.DataFrame(preds_long_rows).to_parquet(DATA_DIR / 'predictions.parquet', index=False)
results_df.to_parquet(DATA_DIR / 'metrics.parquet', index=False)
results_df.to_csv(DATA_DIR / 'metrics.csv', index=False)
tweets_df[["date","text","compound","pos","neg"]].sample(
    n=min(5000, len(tweets_df)), random_state=42
).reset_index(drop=True).to_parquet(DATA_DIR / 'tweets_sample.parquet', index=False)

shutil.make_archive('/content/data_new', 'zip', '/content/data')
print("DONE! Siap didownload.")
\```

4. Download file-file dari folder `data` di panel Files Colab
5. Upload ke folder `data/` di repo GitHub ini
6. Streamlit Cloud akan otomatis redeploy

> **Catatan:** Jangan jalankan training di Streamlit Cloud karena akan timeout dan out-of-memory. Selalu run di Google Colab lalu upload hasilnya ke GitHub.

---

## 🧠 Pipeline Recap

| Stage | Detail |
|---|---|
| **Data BTC** | CSV CoinGecko: 1 Jan 2021 – 31 Dec 2025 (1826 hari) |
| **Tweets** | HuggingFace `StephanAkkerman/financial-tweets-crypto`, filter mention BTC/Bitcoin → 20.539 tweets |
| **Sentimen** | VADER compound + pos + neg, aggregate per hari (hari kosong → 0) |
| **Features** | close, market_cap, volume, sentimen, lag (1/3/7), MA (7/14/30), rolling std (7/14), return (1d/7d) |
| **Targets** | `target_t1` (besok), `target_t30` (30 hari ke depan) |
| **Split** | Train 2021–2023 · Val 2024 · Test 2025 — chronological, no shuffle |
| **Scaler** | StandardScaler, di-fit hanya di train |
| **Models** | LinearRegression · RandomForest (300 trees, depth 10) · LSTM (64→32 hidden, dropout 0.2, window 30) |

**No-leakage compliance:** chronological split, train-only scaler, causal lag/rolling/windowing.

---

## 📊 Hasil Singkat (Test 2025)

| Horizon | Best Model | RMSE | MAE | MAPE |
|---|---|---:|---:|---:|
| t+1 | LinearRegression | ~2 484 | ~1 978 | ~1.95 % |
| t+30 | LinearRegression | ~28 353 | ~26 776 | ~25.87 % |

Linear Regression menang karena efektif kolaps ke baseline naive (`besok ≈ hari ini`),
yang ternyata sulit dikalahkan untuk forecast harga aset volatile seperti BTC.
Random Forest gagal ekstrapolasi keluar range training; LSTM struggle dengan
distribution shift train (peak \\$70k) → test (peak >\\$120k).

---

## 🔧 Troubleshooting

**Error `Error installing requirements` / `pandas-2.2.2.tar.gz` build failed:**
→ Streamlit Cloud default-nya pakai Python 3.14 sekarang, dan beberapa lib (pandas, numpy)
belum punya prebuilt wheel buat 3.14 — pip jadi coba build from source dan gagal.
**Fix:** file `runtime.txt` di repo ini sudah set `python-3.11` — kalau masih error,
buka Streamlit Cloud → klik app → **Settings** → **Advanced settings** → set Python version
ke `3.11` manual, lalu **Reboot app**. Atau push ulang setelah pastikan `runtime.txt` ke-commit.

**App error "FileNotFoundError" di Streamlit Cloud:**
→ Folder `data/` belum ke-push ke GitHub. Pastikan semua file `.parquet` di-commit.

**App slow / timeout:**
→ Pastikan `requirements.txt` (yang ringan) yang dipakai, BUKAN `requirements-train.txt`.

**Mau pakai dataset / ganti period:**
→ Edit `precompute.py` (variabel `BTC_CSV`, `TRAIN_END`, `VAL_END`, dll.) → re-run → push.

---

## 📝 License

MIT — bebas dipakai dan dimodifikasi.
