"""
Train and serialize all ML artifacts for the MADS API.

Reads raw survey responses directly from RespuestasSemillero_completo.json,
computes the 10 derived features, fits a MinMaxScaler on those features,
and trains the four anomaly-detection models.
"""
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from tensorflow import keras

DATA_PATH = os.environ.get("DATA_PATH", "./data")
MODEL_STORE_PATH = os.environ.get("MODEL_STORE_PATH", "./models_store")
DATASET_FILE = "RespuestasSemillero_completo.json"

F_COLS = [
    "F_Average_Performance",
    "F_Academic_Load",
    "F_Life_Balance",
    "F_Psychological_Stress",
    "F_Family_Support",
    "F_Grade_Consistency",
    "F_Responsibility_Result_Index",
    "F_Parental_Education",
    "F_Socioeconomic_Risk",
    "F_Interest_Performance_Gap",
]

Q_COLS_FOR_CLASSIFY = [
    "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9",
    "Q12", "Q13", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20",
    "Q21", "Q23", "Q24", "Q28", "Q29", "Q32", "Q33", "Q34", "Q35",
]

# All Q columns needed for features or classification
Q_COLS_NEEDED = list(dict.fromkeys(Q_COLS_FOR_CLASSIFY + [
    "Q6", "Q7", "Q8", "Q9", "Q12", "Q13", "Q15", "Q16",
    "Q17", "Q18", "Q20", "Q21", "Q32", "Q33", "Q34", "Q35", "Q3",
]))


def load_raw_data(json_path: str) -> pd.DataFrame:
    df = pd.read_json(json_path)
    # Convert all needed Q columns to numeric; text-answer columns become NaN
    for col in Q_COLS_NEEDED:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[Q_COLS_NEEDED].dropna()
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["F_Average_Performance"] = (df["Q33"] + df["Q34"] + df["Q35"]) / 3
    f["F_Academic_Load"] = df["Q7"] + df["Q8"] + df["Q9"]
    f["F_Life_Balance"] = df["Q20"] / (f["F_Academic_Load"] + 1)
    f["F_Psychological_Stress"] = (df["Q17"] + df["Q18"]) / 2
    f["F_Family_Support"] = df["Q12"] + df["Q13"]
    f["F_Grade_Consistency"] = df[["Q33", "Q34", "Q35"]].std(axis=1, ddof=1)
    f["F_Responsibility_Result_Index"] = (
        df["Q3"] / (f["F_Average_Performance"] + 1)
    )
    f["F_Parental_Education"] = df["Q15"] + df["Q16"]
    f["F_Socioeconomic_Risk"] = (6 - df["Q32"]) + (1 - df["Q21"])
    f["F_Interest_Performance_Gap"] = abs(
        df["Q6"] - (f["F_Average_Performance"] / 3)
    )
    return f


def save_pkl(obj, name: str) -> None:
    path = os.path.join(MODEL_STORE_PATH, name)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"  Saved {name}")


def main() -> None:
    os.makedirs(MODEL_STORE_PATH, exist_ok=True)

    # ── Load raw data ──────────────────────────────────────────────────────
    json_path = os.path.join(DATA_PATH, DATASET_FILE)
    print(f"Loading dataset from {json_path} ...")
    df_raw = load_raw_data(json_path)
    print(f"  Shape: {df_raw.shape}")

    df_features = compute_features(df_raw)

    # ── Scaler for 10 F_ features ──────────────────────────────────────────
    print("\n[1/7] Fitting scaler_features.pkl ...")
    X_raw = df_features[F_COLS].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    save_pkl(scaler, "scaler_features.pkl")

    # ── Semillero means and stds (raw Q values) ────────────────────────────
    print("[2/7] Computing semillero_means.pkl and semillero_stds.pkl ...")
    semillero_means = df_raw[Q_COLS_FOR_CLASSIFY].mean().to_dict()
    semillero_stds = df_raw[Q_COLS_FOR_CLASSIFY].std(ddof=1).to_dict()
    save_pkl(semillero_means, "semillero_means.pkl")
    save_pkl(semillero_stds, "semillero_stds.pkl")

    # ── One-Class SVM ──────────────────────────────────────────────────────
    print("[3/7] Training OC-SVM ...")
    ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.10)
    ocsvm.fit(X_scaled)
    save_pkl(ocsvm, "ocsvm.pkl")

    # ── Local Outlier Factor ───────────────────────────────────────────────
    print("[4/7] Training LOF ...")
    lof = LocalOutlierFactor(
        n_neighbors=20, contamination=0.10, novelty=True, n_jobs=-1
    )
    lof.fit(X_scaled)
    save_pkl(lof, "lof.pkl")

    # ── Isolation Forest ───────────────────────────────────────────────────
    print("[5/7] Training Isolation Forest ...")
    iso_forest = IsolationForest(
        contamination=0.10,
        n_estimators=100,
        max_samples="auto",
        max_features=1.0,
        random_state=42,
    )
    iso_forest.fit(X_scaled)
    save_pkl(iso_forest, "iso_forest.pkl")

    # ── Autoencoder ────────────────────────────────────────────────────────
    print("[6/7] Training Autoencoder ...")
    tf.random.set_seed(42)
    input_dim = X_scaled.shape[1]  # 10
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(32, activation="relu")(inputs)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(input_dim, activation="sigmoid")(x)
    autoencoder = keras.Model(inputs, outputs, name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )
    ae_path = os.path.join(MODEL_STORE_PATH, "autoencoder.keras")
    autoencoder.save(ae_path)
    print("  Saved autoencoder.keras")

    train_recon = autoencoder.predict(X_scaled, verbose=0)
    train_mse = np.mean(np.power(X_scaled - train_recon, 2), axis=1)
    threshold = float(np.percentile(train_mse, 95))
    save_pkl(threshold, "autoencoder_threshold.pkl")
    print(f"  Autoencoder threshold (p95): {threshold:.6f}")

    # ── Verify all artifacts load correctly ────────────────────────────────
    print("\n[7/7] Verifying artifacts ...")
    artifacts = [
        "scaler_features.pkl", "semillero_means.pkl", "semillero_stds.pkl",
        "ocsvm.pkl", "lof.pkl", "iso_forest.pkl",
        "autoencoder_threshold.pkl",
    ]
    for name in artifacts:
        with open(os.path.join(MODEL_STORE_PATH, name), "rb") as fh:
            pickle.load(fh)
        print(f"  OK {name}")
    tf.keras.models.load_model(os.path.join(MODEL_STORE_PATH, "autoencoder.keras"))
    print("  OK autoencoder.keras")

    print("\nAll 8 artifacts ready in", MODEL_STORE_PATH)


if __name__ == "__main__":
    main()
