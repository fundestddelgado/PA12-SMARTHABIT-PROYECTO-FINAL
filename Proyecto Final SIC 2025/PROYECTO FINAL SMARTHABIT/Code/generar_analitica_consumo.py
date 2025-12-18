import os
import json
import pandas as pd

# ============================
# ARCHIVOS DE ENTRADA / SALIDA
# ============================
DATASET = r"C:\Users\diego\Desktop\Proyecto SamSung Final\Data Set\dataset_consumo_total_1y.csv"
OUT_FILE = r"C:\Users\diego\Desktop\Proyecto SamSung Final\Dashboard\static\analitica_consumo.json"


# ============================
# 1) CARGAR DATASET COMPLETO
# ============================
def cargar_dataset():
    df = pd.read_csv(DATASET)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    return df


# ============================
# 2) CALCULAR KPIs
# ============================
def calcular_kpis(df):

    # Agrupar por día
    df_day = df.groupby("date").agg({
        "energia_kwh": "sum",
        "agua_litros": "sum",
        "costo_total_usd": "sum"
    })

    kpis = {
        "promedio_energia_diaria": round(df_day["energia_kwh"].mean(), 4),
        "consumo_maximo_diario": round(df_day["energia_kwh"].max(), 4),
        "dia_mas_costoso": str(df_day["costo_total_usd"].idxmax()),
        "costo_maximo_diario": round(df_day["costo_total_usd"].max(), 4)
    }

    return kpis, df_day


# ============================
# 3) SERIES ULTIMOS 7 DÍAS
# ============================
def ultimos_7_dias(df_day):
    last7 = df_day.tail(7)
    return {
        "fechas": [str(i) for i in last7.index],
        "energia": [float(v) for v in last7["energia_kwh"]],
        "agua":    [float(v) for v in last7["agua_litros"]],
        "costo":   [float(v) for v in last7["costo_total_usd"]]
    }


# ============================
# 4) TOP 5 DÍAS MÁS COSTOSOS
# ============================
def top_5_costoso(df_day):
    top = df_day.sort_values("costo_total_usd", ascending=False).head(5)
    return {
        "fechas": [str(i) for i in top.index],
        "costos": [float(c) for c in top["costo_total_usd"]]
    }


# ============================
# 5) PROMEDIO POR HORA DEL DÍA
# ============================
def promedio_por_hora(df):
    df_hour = df.groupby("hour").agg({
        "energia_kwh": "mean",
        "agua_litros": "mean",
        "costo_total_usd": "mean"
    })
    return {
        "hora": list(df_hour.index),
        "energia": [float(v) for v in df_hour["energia_kwh"]],
        "agua":    [float(v) for v in df_hour["agua_litros"]],
        "costo":   [float(v) for v in df_hour["costo_total_usd"]]
    }


# ============================
# 6) GENERAR JSON FINAL
# ============================
def generar_json():
    df = cargar_dataset()

    # KPIs
    kpis, df_day = calcular_kpis(df)

    # Últimos 7 días
    serie_7d = ultimos_7_dias(df_day)

    # Top 5 costosos
    top5 = top_5_costoso(df_day)

    # Promedio por hora
    hora = promedio_por_hora(df)

    salida = {
        "kpis": kpis,
        "ultimos_7_dias": serie_7d,
        "top5_costosos": top5,
        "promedio_hora": hora
    }

    # Exportar JSON
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=4)

    print("Archivo generado correctamente:")
    print(OUT_FILE)


if __name__ == "__main__":
    generar_json()
