import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


# =========================================
# Config
# =========================================
DEFAULT_CSV = r"C:\Users\diego\Desktop\Proyecto SamSung Final\Data Set\dataset_consumo_total_1y.csv"
OUT_DIR = r"C:\Users\diego\Desktop\Proyecto SamSung Final\Pred_vs_Real"

# Ventana y horizontes (en horas)
WINDOW = 168
HORIZONS = [1, 6, 12, 24]  # 4 horizontes
# Targets por horizonte: energía, agua, costo_total → 3 * 4 = 12 salidas
TARGETS = ["energia_kwh", "agua_litros", "costo_total_usd"]
N_TARGETS = len(TARGETS) * len(HORIZONS)  # 12


# =========================================
# Utilidades
# =========================================
def build_sequences(df, feature_cols, target_cols, window=WINDOW, horizons=HORIZONS):
    """
    Crea X (ventanas) y y (salidas multi-horizonte).
    X.shape = (muestras, window, n_features)
    y.shape = (muestras, N_TARGETS) en orden:
       [energia_1h, agua_1h, costo_1h,  energia_6h, agua_6h, costo_6h, ...  energia_24h, agua_24h, costo_24h]
    """
    data = df[feature_cols].values
    # Para reconstruir timestamps y graficar
    timestamps = pd.to_datetime(df["timestamp"]).values

    max_h = max(horizons)
    X, y, base_times = [], [], []

    for i in range(window, len(df) - max_h):
        X.append(data[i - window:i, :])
        # concatenamos targets de cada horizonte en orden
        y_i = []
        for h in horizons:
            for tcol in target_cols:
                y_i.append(df.loc[i + h, tcol])
        y.append(y_i)
        base_times.append(timestamps[i])  # tiempo base que origina la predicción

    return np.array(X), np.array(y), np.array(base_times)


def inverse_targets(pred_scaled, real_scaled, scaler, n_features, feature_order):
    """
    Desescala vectores de salida que fueron escalados con el mismo scaler de features.
    feature_order = lista con el orden de columnas que se escalaron.
    Asumimos que TARGETS son las 3 primeras columnas del feature_order.
    """
    # posiciones de targets en el vector de features
    idx_energy = feature_order.index("energia_kwh")
    idx_water  = feature_order.index("agua_litros")
    idx_cost   = feature_order.index("costo_total_usd")
    idxs = [idx_energy, idx_water, idx_cost]

    def invert_block(block3):
        tmp = np.zeros((block3.shape[0], n_features))
        tmp[:, idxs] = block3
        inv = scaler.inverse_transform(tmp)[:, idxs]
        return inv  # (n, 3) → energia, agua, costo

    # pred_scaled: (n, 12) => 4 bloques de 3
    # real_scaled: (n, 12)
    pred_blocks, real_blocks = [], []
    for k in range(0, pred_scaled.shape[1], 3):
        pred_blk = pred_scaled[:, k:k+3]
        real_blk = real_scaled[:, k:k+3]
        pred_blocks.append(invert_block(pred_blk))
        real_blocks.append(invert_block(real_blk))

    # lista de 4 arrays (n,3) cada una
    return pred_blocks, real_blocks


def metrics_mae_mape(y_true, y_pred, eps=1e-8):
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0
    return mae, mape


# =========================================
# Main
# =========================================
def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    # -----------------------------
    # 1) Cargar dataset
    # -----------------------------
    print("📦 Cargando dataset…")
    df = pd.read_csv(args.csv)

    # Asegurar orden temporal
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------
    # 2) Definir columnas (Opción 2)
    # -----------------------------
    # FEATURES = targets históricos + calendario
    feature_cols = [
        "energia_kwh", "agua_litros", "costo_total_usd",
        "hour", "dayofweek", "month", "is_weekend"
    ]
    target_cols = ["energia_kwh", "agua_litros", "costo_total_usd"]

    # -----------------------------
    # 3) Escalado (fit SOLO con train para evitar fuga)
    #    Split temporal: 80% train, 20% test
    # -----------------------------
    n = len(df)
    split_idx = int(n * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()

    scaler = MinMaxScaler()
    scaler.fit(df_train[feature_cols].values)

    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols].values)

    # -----------------------------
    # 4) Secuencias multi-horizonte
    # -----------------------------
    print("🧩 Construyendo secuencias… (ventana=168, horizontes=1,6,12,24)")
    X, y, base_times = build_sequences(df_scaled, feature_cols, target_cols,
                                       window=args.window, horizons=HORIZONS)

    # Reconstruir split en términos de secuencias:
    # el primer X usable empieza en índice "window"
    # buscamos el índice de corte equivalente en X,y
    first_idx = args.window
    # última posición de secuencia cuyo target no excede el dataset
    last_idx = len(df_scaled) - max(HORIZONS) - 1
    # índice de corte en base a split temporal
    cut = max(0, split_idx - first_idx)
    cut = min(cut, len(X))  # seguridad

    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]
    times_train, times_test = base_times[:cut], base_times[cut:]

    n_features = X.shape[2]

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")

    # -----------------------------
    # 5) Modelo LSTM multi-salida
    # -----------------------------
    print("🧠 Construyendo el modelo…")
    model = Sequential([
        Input(shape=(args.window, n_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(64, activation="relu"),
        Dense(N_TARGETS)  # 12 salidas
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # -----------------------------
    # 6) Entrenamiento
    # -----------------------------
    print("🚀 Entrenando…")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=32,
        callbacks=[es],
        verbose=1
    )

    # Guardar modelo
    model_path = os.path.join(OUT_DIR, "modelo_multihorizonte.keras")
    model.save(model_path)
    print(f"💾 Modelo guardado en: {model_path}")

    # -----------------------------
    # 7) Predicciones y desescalado
    # -----------------------------
    print("🔎 Prediciendo…")
    y_pred = model.predict(X_test, verbose=0)

    # Desescalar: se necesita armar bloques de 3 en el orden de features
    pred_blocks, real_blocks = inverse_targets(
        y_pred, y_test, scaler,
        n_features=n_features,
        feature_order=feature_cols
    )
    # pred_blocks / real_blocks: lista con 4 arrays (n_test, 3) en el orden de HORIZONS

    # -----------------------------
    # 8) Métricas y gráficas por horizonte
    # -----------------------------
    print("📈 Generando gráficas y métricas…")

    # Para cada horizonte y variable (energia, agua, costo) crear gráfica y métricas
    json_rows = []
    horizon_labels = {1: "1h", 6: "6h", 12: "12h", 24: "24h"}

    for i_h, h in enumerate(HORIZONS):
        real_h = real_blocks[i_h]   # (n, 3) -> [energia, agua, costo]
        pred_h = pred_blocks[i_h]

        # Métricas por variable
        var_names = ["energia_kwh", "agua_litros", "costo_total_usd"]
        for j, var in enumerate(var_names):
            mae, mape = metrics_mae_mape(real_h[:, j], pred_h[:, j])
            print(f"H{h:>2} - {var:16s} | MAE={mae:.4f} | MAPE={mape:.2f}%")

            # Gráfica
            plt.figure(figsize=(12, 5))
            plt.plot(real_h[:, j], label=f"Real {var}")
            plt.plot(pred_h[:, j], label=f"Pred {var}")
            plt.title(f"{var} – Real vs Pred ({horizon_labels[h]})")
            plt.legend()
            fname = f"pred_vs_real_{var}_{horizon_labels[h]}.png"
            out_path = os.path.join(OUT_DIR, fname)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()

        # Agregar filas al JSON para este horizonte (tomamos hasta 200 puntos)
        n_take = min(200, real_h.shape[0])
        for k in range(n_take):
            # timestamp objetivo = tiempo base + h horas
            target_time = pd.to_datetime(times_test[k]) + pd.to_timedelta(h, unit="h")
            json_rows.append({
                "horizon": h,
                "timestamp_objetivo": target_time.strftime("%Y-%m-%d %H:%M:%S"),
                "energia_real_kwh": float(real_h[k, 0]),
                "energia_pred_kwh": float(pred_h[k, 0]),
                "agua_real_l": float(real_h[k, 1]),
                "agua_pred_l": float(pred_h[k, 1]),
                "costo_real_usd": float(real_h[k, 2]),
                "costo_pred_usd": float(pred_h[k, 2])
            })

    # Guardar histórico de entrenamiento
    plt.figure(figsize=(10, 4))
    plt.plot(hist.history["loss"], label="loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.title("Historial de entrenamiento")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "historial_entrenamiento.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # 9) Guardar JSON para dashboard
    # -----------------------------
    json_path = os.path.join(OUT_DIR, "predicciones_multihorizonte.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2, ensure_ascii=False)
    print(f"💾 JSON guardado en: {json_path}")

    print("\n✅ Proceso COMPLETO.")


# =========================================
# CLI
# =========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    main(args)
   