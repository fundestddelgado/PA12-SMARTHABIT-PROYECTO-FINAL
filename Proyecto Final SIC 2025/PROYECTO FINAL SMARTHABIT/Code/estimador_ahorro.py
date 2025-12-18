import os
import json
import pandas as pd

# --------------------------------
# ARCHIVOS
# --------------------------------
DATASET = r"C:\Users\diego\Desktop\Proyecto SamSung Final\Data Set\dataset_consumo_total_1y.csv"
PRED_JSON = r"C:\Users\diego\Desktop\Proyecto SamSung Final\Dashboard\static\predicciones_multihorizonte.json"

OUT_FILE = r"C:\Users\diego\Desktop\Proyecto SamSung Final\Pred_vs_Real\estimacion_ahorro.json"


# --------------------------------
# 1. CARGAR DATASET BASE (REAL)
# --------------------------------
def cargar_dataset(path):
    df = pd.read_csv(path)
    return df


# --------------------------------
# 2. CALCULAR AHORRO REAL POSIBLE
# --------------------------------
def calcular_ahorros_reales(df):

    total_real = df["costo_total_usd"].sum()

    escenarios = {
        "consumo_real_total_usd": round(total_real, 4),
        "ahorro_10pct_usd": round(total_real * 0.10, 4),
        "ahorro_20pct_usd": round(total_real * 0.20, 4),
        "ahorro_30pct_usd": round(total_real * 0.30, 4),
    }

    return escenarios


# --------------------------------
# 3. CARGAR PREDICCIONES MULTIHORIZONTE
# --------------------------------
def cargar_predicciones(path_json):

    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data    # LISTA con filas independientes


# --------------------------------
# 4. CALCULAR AHORRO BASADO EN PREDICCIONES
# --------------------------------
def calcular_ahorro_predicho(lista_pred):

    df = pd.DataFrame(lista_pred)

    resultados = []

    for h in sorted(df["horizon"].unique()):
        df_h = df[df["horizon"] == h]

        # Costo futuro predicho
        costo_total_pred = df_h["costo_pred_usd"].sum()

        resultados.append({
            "horizonte_horas": int(h),
            "costo_estimado_usd": round(costo_total_pred, 4),
            "ahorro_10pct_usd": round(costo_total_pred * 0.10, 4),
            "ahorro_20pct_usd": round(costo_total_pred * 0.20, 4),
            "ahorro_30pct_usd": round(costo_total_pred * 0.30, 4),
        })

    return resultados


# --------------------------------
# 5. EXPORTAR RESULTADOS
# --------------------------------
def guardar_json(real, pred):
    salida = {
        "ahorro_actual_posible": real,
        "ahorro_predicho_futuro": pred
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=4)

    print(f"\nOK → Archivo generado:\n{OUT_FILE}")


# --------------------------------
# MAIN
# --------------------------------
def main():

    print("Cargando dataset completo…")
    df = cargar_dataset(DATASET)

    print("Calculando escenarios de ahorro real…")
    ahorro_real = calcular_ahorros_reales(df)

    print("Cargando predicciones multihorizonte…")
    pred = cargar_predicciones(PRED_JSON)

    print("Calculando ahorro basado en predicción…")
    ahorro_pred = calcular_ahorro_predicho(pred)

    guardar_json(ahorro_real, ahorro_pred)


if __name__ == "__main__":
    main()
