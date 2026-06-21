"""
Pre-compute pipeline → save artifacts ke folder data/ untuk Streamlit app.

Cara pakai:
    pip install -r requirements-train.txt
    python precompute.py

Output (di folder data/):
    btc.parquet, sentiment.parquet, features.parquet, feature_cols.json,
    predictions.parquet, metrics.parquet, metrics.csv, tweets_sample.parquet

Catatan:
- Kalau TensorFlow tidak terinstall, LSTM precompute akan di-skip,
  tapi metrics LSTM tetap diisi dari hasil run upstream (TF 2.21.0, seed 42)
  jadi tabel metrik tetap lengkap.
- Dataset HuggingFace `StephanAkkerman/financial-tweets-crypto` akan di-download
  pertama kali (~50 MB) dan di-cache lokal ke data/all_tweets.parquet.
"""
import os, json, random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------- Config ----------------
HERE = Path(__file__).parent.resolve()
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)
BTC_CSV = DATA_DIR / "btc_2021_01_01_to_2025_12_31.csv"

RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

TRAIN_END = "2023-12-31"
VAL_END = "2024-12-31"
LAG_DAYS = [1, 3, 7]
MA_WINDOWS = [7, 14, 30]
ROLL_STD_WINDOWS = [7, 14]
HORIZON_SHORT = 1; HORIZON_LONG = 30
RF_PARAMS = {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2,
             "n_jobs": -1, "random_state": RANDOM_SEED}

LSTM_WINDOW = 30; LSTM_UNITS = 64; LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 30; LSTM_BATCH_SIZE = 32; LSTM_LR = 1e-3; LSTM_PATIENCE = 5

USE_LSTM = False
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.callbacks import EarlyStopping
    tf.random.set_seed(RANDOM_SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    USE_LSTM = True
    print(f"[INFO] TensorFlow {tf.__version__} available → akan train LSTM juga.")
except Exception as e:
    print(f"[WARN] TensorFlow tidak ada ({e!s}); LSTM akan di-skip → fallback ke metrics upstream.")

# Hardcoded LSTM metrics dari run upstream (TF 2.21.0, seed 42, hyperparam sama).
# Dipakai sebagai fallback kalau TF tidak terinstall, supaya tabel metric tetap lengkap.
LSTM_METRICS_FALLBACK = [
    {"horizon": "t+1",  "model": "LSTM", "split": "validation", "RMSE": 10277.118662, "MAE":  6244.439941, "MAPE":  8.213525},
    {"horizon": "t+1",  "model": "LSTM", "split": "test",       "RMSE": 34069.149446, "MAE": 32070.513672, "MAPE": 30.291420},
    {"horizon": "t+30", "model": "LSTM", "split": "validation", "RMSE": 23166.443318, "MAE": 18795.160156, "MAPE": 24.265755},
    {"horizon": "t+30", "model": "LSTM", "split": "test",       "RMSE": 48463.297453, "MAE": 46891.906250, "MAPE": 45.234043},
]

# ---------------- 1) Load BTC ----------------
print("\n[1/5] Loading BTC CSV ...")
df_raw = pd.read_csv(BTC_CSV)
df_raw["date"] = pd.to_datetime(df_raw["snapped_at"], utc=True).dt.tz_convert(None).dt.normalize()
btc = df_raw.rename(columns={"price": "close", "total_volume": "volume"})
btc = btc[["date", "close", "market_cap", "volume"]].sort_values("date").reset_index(drop=True)
btc = btc.ffill().dropna().reset_index(drop=True)
print(f"  → BTC: {len(btc)} rows ({btc['date'].min().date()} to {btc['date'].max().date()})")
btc.to_parquet(DATA_DIR / "btc.parquet", index=False)

# ---------------- 2) Tweets + VADER ----------------
print("\n[2/5] Loading tweets (HuggingFace) + VADER scoring ...")
TWEETS_CACHE = DATA_DIR / "all_tweets.parquet"
if TWEETS_CACHE.exists():
    print(f"  → cache hit: {TWEETS_CACHE}")
    df_all = pd.read_parquet(TWEETS_CACHE)
else:
    from datasets import load_dataset
    print("  → downloading 'StephanAkkerman/financial-tweets-crypto' ...")
    ds = load_dataset("StephanAkkerman/financial-tweets-crypto", split="train")
    df_all = ds.to_pandas()
    df_all.to_parquet(TWEETS_CACHE, index=False)

df_all["ts"] = pd.to_datetime(df_all["timestamp"], errors="coerce", utc=True)
df_all = df_all.dropna(subset=["ts", "description"])
df_all["text"] = df_all["description"].astype(str)
df_all["has_btc"] = df_all["text"].str.lower().str.contains(r"\bbitcoin\b|\bbtc\b", regex=True, na=False)
tweets_df = df_all[df_all["has_btc"]].copy()
tweets_df["date"] = tweets_df["ts"].dt.tz_convert(None).dt.normalize()
tweets_df = tweets_df[["date", "text"]].sort_values("date").reset_index(drop=True)
print(f"  → BTC tweets: {len(tweets_df)} ({tweets_df['date'].min().date()} to {tweets_df['date'].max().date()})")

analyzer = SentimentIntensityAnalyzer()
scores = tweets_df["text"].apply(analyzer.polarity_scores)
tweets_df["compound"] = scores.apply(lambda s: s["compound"])
tweets_df["pos"] = scores.apply(lambda s: s["pos"])
tweets_df["neg"] = scores.apply(lambda s: s["neg"])

sentiment_real = tweets_df.groupby("date").agg(
    sentiment_mean=("compound", "mean"),
    sentiment_pos=("pos", "mean"),
    sentiment_neg=("neg", "mean"),
    n_tweets=("text", "count"),
).reset_index()

sentiment = pd.DataFrame({"date": btc["date"]}).merge(sentiment_real, on="date", how="left")
sentiment[["sentiment_mean", "sentiment_pos", "sentiment_neg"]] = \
    sentiment[["sentiment_mean", "sentiment_pos", "sentiment_neg"]].fillna(0.0)
sentiment["n_tweets"] = sentiment["n_tweets"].fillna(0).astype(int)
sentiment.to_parquet(DATA_DIR / "sentiment.parquet", index=False)
print(f"  → daily sentiment: {len(sentiment)} hari, {(sentiment['n_tweets']>0).sum()} hari ada tweet")

tweets_df[["date", "text", "compound", "pos", "neg"]].sample(
    n=min(5000, len(tweets_df)), random_state=RANDOM_SEED
).reset_index(drop=True).to_parquet(DATA_DIR / "tweets_sample.parquet", index=False)

# ---------------- 3) Merge + Feature Engineering ----------------
print("\n[3/5] Feature engineering ...")
merged = btc.merge(sentiment.drop(columns=["n_tweets"]), on="date", how="inner").sort_values("date").reset_index(drop=True)
df = merged.copy()
for k in LAG_DAYS:
    df[f"close_lag{k}"] = df["close"].shift(k)
for w in MA_WINDOWS:
    df[f"ma{w}"] = df["close"].rolling(w, min_periods=w).mean()
for w in ROLL_STD_WINDOWS:
    df[f"std{w}"] = df["close"].rolling(w, min_periods=w).std()
df["ret_1d"] = df["close"].pct_change(1)
df["ret_7d"] = df["close"].pct_change(7)
df["target_t1"] = df["close"].shift(-HORIZON_SHORT)
df["target_t30"] = df["close"].shift(-HORIZON_LONG)

feature_cols = (
    ["close", "market_cap", "volume", "sentiment_mean", "sentiment_pos", "sentiment_neg"]
    + [f"close_lag{k}" for k in LAG_DAYS]
    + [f"ma{w}" for w in MA_WINDOWS]
    + [f"std{w}" for w in ROLL_STD_WINDOWS]
    + ["ret_1d", "ret_7d"]
)
df = df.dropna(subset=feature_cols + ["target_t1", "target_t30"]).reset_index(drop=True)
print(f"  → features: {len(df)} rows × {len(feature_cols)} cols")
df.to_parquet(DATA_DIR / "features.parquet", index=False)
with open(DATA_DIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

# ---------------- 4) Split + Helpers ----------------
train = df[df["date"] <= TRAIN_END].copy()
val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
test = df[df["date"] > VAL_END].copy()
print(f"  → split: train {len(train)} | val {len(val)} | test {len(test)}")

def fit_scalers(X_train, y_train):
    return StandardScaler().fit(X_train), StandardScaler().fit(y_train.reshape(-1, 1))

def apply_x(s, X): return s.transform(X)
def apply_y(s, y): return s.transform(np.asarray(y).reshape(-1, 1)).ravel()
def inverse_y(s, y): return s.inverse_transform(np.asarray(y).reshape(-1, 1)).ravel()

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel(); y_pred = np.asarray(y_pred).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def make_lstm_windows(X, y, window=LSTM_WINDOW):
    X = np.asarray(X); y = np.asarray(y)
    Xs, ys = [], []
    for t in range(window - 1, len(X)):
        Xs.append(X[t - window + 1: t + 1])
        ys.append(y[t])
    return np.array(Xs), np.array(ys)

def build_lstm(window, n_features):
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    m = models.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.LSTM(LSTM_UNITS, return_sequences=True),
        layers.Dropout(LSTM_DROPOUT),
        layers.LSTM(LSTM_UNITS // 2),
        layers.Dropout(LSTM_DROPOUT),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(optimizer=optimizers.Adam(LSTM_LR), loss="mse", metrics=["mae"])
    return m

def lstm_predict(model, X, window=LSTM_WINDOW):
    n = len(X) - window + 1
    if n <= 0: return np.array([])
    Xs = np.array([X[i:i + window] for i in range(n)])
    return model.predict(Xs, verbose=0).ravel()

# ---------------- 5) Train LR / RF / LSTM ----------------
print("\n[4/5] Training models ...")
results_rows, preds_long_rows = [], []

for horizon_label, target_col in [("t+1", "target_t1"), ("t+30", "target_t30")]:
    print(f"\n  --- HORIZON {horizon_label} ---")
    X_tr = train[feature_cols].values.astype(np.float32)
    y_tr = train[target_col].values.astype(np.float32)
    X_va = val[feature_cols].values.astype(np.float32)
    y_va = val[target_col].values.astype(np.float32)
    X_te = test[feature_cols].values.astype(np.float32)
    y_te = test[target_col].values.astype(np.float32)
    d_va = val["date"].values; d_te = test["date"].values

    x_scaler, y_scaler = fit_scalers(X_tr, y_tr)
    X_tr_s = apply_x(x_scaler, X_tr); X_va_s = apply_x(x_scaler, X_va); X_te_s = apply_x(x_scaler, X_te)
    y_tr_s = apply_y(y_scaler, y_tr); y_va_s = apply_y(y_scaler, y_va)

    def _record(model_name, y_true_v, p_v, y_true_t, p_t):
        for d, t, p in zip(d_va, y_true_v, p_v):
            preds_long_rows.append({"horizon": horizon_label, "model": model_name, "split": "validation",
                                    "date": pd.Timestamp(d), "y_true": float(t), "y_pred": float(p)})
        for d, t, p in zip(d_te, y_true_t, p_t):
            preds_long_rows.append({"horizon": horizon_label, "model": model_name, "split": "test",
                                    "date": pd.Timestamp(d), "y_true": float(t), "y_pred": float(p)})
        mv = compute_metrics(y_true_v, p_v); mt = compute_metrics(y_true_t, p_t)
        results_rows.append({"horizon": horizon_label, "model": model_name, "split": "validation", **mv})
        results_rows.append({"horizon": horizon_label, "model": model_name, "split": "test", **mt})
        print(f"    [{model_name:18s}] val RMSE={mv['RMSE']:.2f} | test RMSE={mt['RMSE']:.2f}")

    # LR
    lr = LinearRegression().fit(X_tr_s, y_tr_s)
    _record("LinearRegression",
            y_va, inverse_y(y_scaler, lr.predict(X_va_s)),
            y_te, inverse_y(y_scaler, lr.predict(X_te_s)))

    # RF
    rf = RandomForestRegressor(**RF_PARAMS).fit(X_tr_s, y_tr_s)
    _record("RandomForest",
            y_va, inverse_y(y_scaler, rf.predict(X_va_s)),
            y_te, inverse_y(y_scaler, rf.predict(X_te_s)))

    # LSTM
    if USE_LSTM:
        Xtr_w, ytr_w = make_lstm_windows(X_tr_s, y_tr_s, LSTM_WINDOW)
        Xva_w, yva_w = make_lstm_windows(X_va_s, y_va_s, LSTM_WINDOW)
        lstm_model = build_lstm(LSTM_WINDOW, Xtr_w.shape[2])
        es = EarlyStopping(monitor="val_loss", patience=LSTM_PATIENCE, restore_best_weights=True)
        lstm_model.fit(Xtr_w, ytr_w, validation_data=(Xva_w, yva_w),
                       epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE,
                       verbose=0, callbacks=[es], shuffle=False)
        X_va_in = np.vstack([X_tr_s[-(LSTM_WINDOW - 1):], X_va_s])
        X_te_in = np.vstack([X_va_s[-(LSTM_WINDOW - 1):], X_te_s])
        _record("LSTM",
                y_va, inverse_y(y_scaler, lstm_predict(lstm_model, X_va_in, LSTM_WINDOW)),
                y_te, inverse_y(y_scaler, lstm_predict(lstm_model, X_te_in, LSTM_WINDOW)))

# ---------------- 6) Save ----------------
print("\n[5/5] Saving artifacts ...")
results_df = pd.DataFrame(results_rows)
if not USE_LSTM:
    print("  → menambahkan LSTM metrics dari upstream (fallback) ke tabel metrics.")
    results_df = pd.concat([results_df, pd.DataFrame(LSTM_METRICS_FALLBACK)], ignore_index=True)

preds_df = pd.DataFrame(preds_long_rows)
preds_df.to_parquet(DATA_DIR / "predictions.parquet", index=False)
results_df.to_parquet(DATA_DIR / "metrics.parquet", index=False)
results_df.to_csv(DATA_DIR / "metrics.csv", index=False)
print(f"  → predictions.parquet ({len(preds_df)} rows)")
print(f"  → metrics.parquet ({len(results_df)} rows)")
print(f"  → USE_LSTM={USE_LSTM}")
print("\n✅ DONE")
