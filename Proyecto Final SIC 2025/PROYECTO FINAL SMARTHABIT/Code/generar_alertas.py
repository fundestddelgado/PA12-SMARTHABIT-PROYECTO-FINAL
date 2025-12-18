# -*- coding: utf-8 -*-
"""
Módulo de Alertas Inteligentes (energía/agua/costo) para el dashboard.

Lee:
  - Data Set/dataset_consumo_total_1y.csv     (baseline y tarifas)
  - Pred_vs_Real/predicciones_multihorizonte.json

Genera:
  - Pred_vs_Real/alertas_multihorizonte.json

Autor: Bro (Proyecto Samsung Final)
"""

import os
import json
import math
import uuid
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# 1) CONFIGURACIÓN
# =========================
BASE_DIR = r"C:\Users\diego\Desktop\Proyecto SamSung Final"
DATA_DIR = os.path.join(BASE_DIR, "Data Set")
PRED_DIR = os.path.join(BASE_DIR, "Pred_vs_Real")

DATASET_CSV = os.path.join(DATA_DIR, "dataset_consumo_total_1y.csv")
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PRED_JSON = os.path.join(BASE_DIR, "Dashboard", "static", "predicciones_multihorizonte.json")
OUT_ALERTAS_JSON = os.path.join(PRED_DIR, "alertas_multihorizonte.json")

# Presupuesto mensual (puedes ajustarlo en USD)
PRESUPUESTO_MENSUAL_USD = 120.0

# Umbrales (fáciles de ajustar)
CFG = {
    "umbral_percentil": 0.95,      # p95 para picos
    "factor_exceso": 1.15,         # pico si pred > p95 * 1.15
    "trend_window_days": 7,         # media móvil días
    "trend_increase_pct": 0.20,     # +20% vs media previa => alerta
    "leak_hours": [1, 2, 3, 4],     # horas nocturnas
    "leak_factor": 1.50,            # litros nocturnos > baseline*1.5
    "min_samples_baseline": 200,    # robustez para baseline
}


# =========================
# 2) UTILIDADES
# =========================
def _safe_dt(x) -> pd.Timestamp:
    if pd.isna(x):
        return pd.NaT
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT


def _severity(pct_over: float) -> str:
    """Clasifica severidad con base en % sobre umbral."""
    if pct_over >= 0.50:
        return "crítica"
    if pct_over >= 0.25:
        return "moderada"
    return "leve"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# =========================
# 3) CARGA DE DATOS
# =========================
def cargar_dataset_base(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    # Normaliza timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("El dataset base no tiene columna 'timestamp'.")

    # Garantiza columnas clave
    requeridas = {"energia_kwh", "agua_litros", "costo_total_usd"}
    faltan = requeridas - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en dataset base: {faltan}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["ym"] = df["timestamp"].dt.to_period("M").astype(str)
    return df


def cargar_predicciones(path_json: str) -> pd.DataFrame:
    """
    Acepta dos formatos:
      A) dict por horizonte: {"1h": [...], "6h": [...], ...}
      B) lista plana: [{"timestamp": ..., "horizonte": ..., "pred_energia"...}]

    Devuelve DataFrame con columnas:
      timestamp, horizonte, pred_energia_kwh, pred_agua_litros, pred_costo_usd
      (si hay reales, también real_*)
    """
    with open(path_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    registros = []
    if isinstance(raw, dict):
        for hz, rows in raw.items():
            for r in rows:
                registros.append(_normalizar_registro(r, hz))
    elif isinstance(raw, list):
        for r in raw:
            hz = r.get("horizonte") or r.get("horizon") or "1h"
            registros.append(_normalizar_registro(r, hz))
    else:
        raise ValueError("Formato de predicciones no reconocido.")

    dfp = pd.DataFrame(registros)
    # Limpieza básica
    dfp["timestamp"] = pd.to_datetime(dfp["timestamp"])
    dfp = dfp.sort_values(["horizonte", "timestamp"]).reset_index(drop=True)
    return dfp


def _normalizar_registro(r: Dict, horizonte: str) -> Dict:
    """
    Intenta mapear diferentes nombres de campos a uno estándar.
    """
    ts = r.get("timestamp") or r.get("ts") or r.get("time")
    # Predicciones
    p_e = r.get("pred_energia") or r.get("pred_energia_kwh") or r.get("energia_pred") or r.get("energy_pred")
    p_a = r.get("pred_agua") or r.get("pred_agua_litros") or r.get("agua_pred") or r.get("water_pred")
    p_c = r.get("pred_costo") or r.get("pred_costo_total_usd") or r.get("costo_pred") or r.get("cost_pred")
    # Reales (si existen)
    r_e = r.get("real_energia") or r.get("real_energia_kwh") or r.get("energia_real") or r.get("energy_real")
    r_a = r.get("real_agua") or r.get("real_agua_litros") or r.get("agua_real") or r.get("water_real")
    r_c = r.get("real_costo") or r.get("real_costo_total_usd") or r.get("costo_real") or r.get("cost_real")

    return {
        "timestamp": _safe_dt(ts),
        "horizonte": str(horizonte),
        "pred_energia_kwh": _to_float(p_e),
        "pred_agua_litros": _to_float(p_a),
        "pred_costo_usd": _to_float(p_c),
        "real_energia_kwh": _to_float(r_e),
        "real_agua_litros": _to_float(r_a),
        "real_costo_usd": _to_float(r_c),
    }


def _to_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


# =========================
# 4) BASELINES / UMBRALES
# =========================
def construir_baselines(df_base: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calcula percentiles por hora/día para energía, agua y costo.
    Devuelve dict con dataframes de umbrales.
    """
    out = {}

    def _p95(s):  # robusto ante NaN
        return np.nanpercentile(s.dropna(), int(CFG["umbral_percentil"] * 100)) if s.notna().sum() > 0 else np.nan

    # por hora (todos los días)
    g_hour = df_base.groupby("hour")
    umbral_hour = pd.DataFrame({
        "p95_energia": g_hour["energia_kwh"].apply(_p95),
        "p95_agua": g_hour["agua_litros"].apply(_p95),
        "p95_costo": g_hour["costo_total_usd"].apply(_p95),
        "median_agua": g_hour["agua_litros"].median(),
    }).reset_index()
    out["por_hora"] = umbral_hour

    # por día de semana (0-6) y hora (mejor ajuste)
    df_base["dow"] = df_base["timestamp"].dt.dayofweek
    g_dow_hour = df_base.groupby(["dow", "hour"])
    umbral_dow_hour = pd.DataFrame({
        "p95_energia": g_dow_hour["energia_kwh"].apply(_p95),
        "p95_agua": g_dow_hour["agua_litros"].apply(_p95),
        "p95_costo": g_dow_hour["costo_total_usd"].apply(_p95),
        "median_agua": g_dow_hour["agua_litros"].median(),
    }).reset_index()
    out["por_dow_hora"] = umbral_dow_hour

    return out


def get_umbral(umbral_df: pd.DataFrame, dow: int, hour: int, var: str) -> float:
    """
    Toma el mejor umbral disponible: por (dow,hour) si existe, si no por hora.
    var ∈ {"energia", "agua", "costo"} → usa p95_var.
    """
    col = f"p95_{var}"
    # por dow/hora
    row = umbral_df["por_dow_hora"]
    df_dh = row[(row["dow"] == dow) & (row["hour"] == hour)]
    if len(df_dh) and not pd.isna(df_dh.iloc[0][col]):
        return float(df_dh.iloc[0][col])

    # fallback por hora
    row_h = umbral_df["por_hora"]
    df_h = row_h[row_h["hour"] == hour]
    if len(df_h) and not pd.isna(df_h.iloc[0][col]):
        return float(df_h.iloc[0][col])

    return np.nan


def get_median_agua_noche(umbral_df: pd.DataFrame, hour: int, dow: int) -> float:
    # similar a get_umbral pero para 'median_agua'
    # (busco primero por dow/hora, luego por hora)
    df_dh = umbral_df["por_dow_hora"]
    sub = df_dh[(df_dh["dow"] == dow) & (df_dh["hour"] == hour)]
    if len(sub) and not pd.isna(sub.iloc[0]["median_agua"]):
        return float(sub.iloc[0]["median_agua"])

    df_h = umbral_df["por_hora"]
    subh = df_h[df_h["hour"] == hour]
    if len(subh) and not pd.isna(subh.iloc[0]["median_agua"]):
        return float(subh.iloc[0]["median_agua"])

    return 0.0


# =========================
# 5) GENERACIÓN DE ALERTAS
# =========================
def generar_alertas(df_base: pd.DataFrame, df_pred: pd.DataFrame) -> List[Dict]:
    umbrales = construir_baselines(df_base)

    alertas: List[Dict] = []

    # --- Preparaciones auxiliares ---
    df_pred = df_pred.copy()
    df_pred["dow"] = df_pred["timestamp"].dt.dayofweek
    df_pred["hour"] = df_pred["timestamp"].dt.hour
    df_pred["ym"] = df_pred["timestamp"].dt.to_period("M").astype(str)

    # 5.1 Picos vs p95 * factor
    for var, col_pred, unidad in [
        ("energia", "pred_energia_kwh", "kWh"),
        ("agua", "pred_agua_litros", "L"),
        ("costo", "pred_costo_usd", "USD"),
    ]:
        for i, row in df_pred[["timestamp", "horizonte", "dow", "hour", col_pred]].dropna().iterrows():
            ts = row["timestamp"]; hz = row["horizonte"]; dow = int(row["dow"]); hour = int(row["hour"])
            val = float(row[col_pred])

            umbral = get_umbral(umbrales, dow, hour, var)
            if pd.isna(umbral) or umbral <= 0:
                continue
            limite = umbral * CFG["factor_exceso"]
            if val > limite:
                pct = (val - limite) / max(limite, 1e-9)
                alertas.append({
                    "id": _new_id(),
                    "tipo": "pico",
                    "variable": var,
                    "unidad": unidad,
                    "timestamp": ts.isoformat(),
                    "horizonte": hz,
                    "valor": round(val, 4),
                    "umbral": round(limite, 4),
                    "porcentaje_sobre_umbral": round(pct * 100, 2),
                    "severidad": _severity(pct),
                    "mensaje": f"Pico de {var} previsto: {val:.2f} {unidad} (> umbral {limite:.2f}).",
                    "recomendacion": _recomendacion_var(var),
                })

    # 5.2 Fuga de agua nocturna
    noct = df_pred[df_pred["hour"].isin(CFG["leak_hours"])].copy()
    for i, row in noct[["timestamp", "horizonte", "dow", "hour", "pred_agua_litros"]].dropna().iterrows():
        ts = row["timestamp"]; hz = row["horizonte"]; dow = int(row["dow"]); hour = int(row["hour"])
        val = float(row["pred_agua_litros"])
        med = get_median_agua_noche(umbrales, hour, dow)
        if med <= 0:
            med = 0.1  # baseline mínimo para evitar divisiones por cero
        limite = med * CFG["leak_factor"]
        if val > limite and val > 1.0:  # evita falsos positivos
            pct = (val - limite) / max(limite, 1e-9)
            alertas.append({
                "id": _new_id(),
                "tipo": "posible_fuga",
                "variable": "agua",
                "unidad": "L",
                "timestamp": ts.isoformat(),
                "horizonte": hz,
                "valor": round(val, 3),
                "umbral": round(limite, 3),
                "porcentaje_sobre_umbral": round(pct * 100, 2),
                "severidad": _severity(pct),
                "mensaje": f"Consumo nocturno de agua inusual: {val:.2f} L entre la(s) {hour}h.",
                "recomendacion": "Verifica grifos/inodoros con fugas y cierre de válvulas por la noche.",
            })

    # 5.3 Overbudget mensual (con predicciones)
    # Suma por mes de pred_costo_usd y compara con presupuesto
    cost_month = df_pred.groupby(["ym", "horizonte"])["pred_costo_usd"].sum(min_count=1).dropna()
    for (ym, hz), total in cost_month.items():
        if total > PRESUPUESTO_MENSUAL_USD:
            pct = (total - PRESUPUESTO_MENSUAL_USD) / PRESUPUESTO_MENSUAL_USD
            alertas.append({
                "id": _new_id(),
                "tipo": "presupuesto_superado",
                "variable": "costo",
                "unidad": "USD",
                "timestamp": pd.Timestamp(f"{ym}-01").isoformat(),
                "horizonte": hz,
                "valor": round(total, 2),
                "umbral": round(PRESUPUESTO_MENSUAL_USD, 2),
                "porcentaje_sobre_umbral": round(pct * 100, 2),
                "severidad": _severity(pct),
                "mensaje": f"Proyección de costo mensual {ym} = ${total:.2f} > presupuesto ${PRESUPUESTO_MENSUAL_USD:.2f}.",
                "recomendacion": "Ajusta hábitos/horarios y revisa grandes consumidores (AC, lavadora, ducha caliente).",
            })

    # 5.4 Tendencia alcista (media móvil 7 días) usando DATASET BASE (reales)
    df_d = df_base[["timestamp", "energia_kwh", "agua_litros", "costo_total_usd"]].copy().set_index("timestamp").resample("1H").sum()
    for var, unidad in [("energia_kwh", "kWh"), ("agua_litros", "L"), ("costo_total_usd", "USD")]:
        s = df_d[var].dropna()
        if len(s) < 24 * (CFG["trend_window_days"] + 1):
            continue
        ma_prev = s.rolling(f"{CFG['trend_window_days']}D").mean().shift(24).dropna()
        ma_now = s.rolling(f"{CFG['trend_window_days']}D").mean().dropna()
        common_idx = ma_prev.index.intersection(ma_now.index)
        inc = (ma_now.loc[common_idx] - ma_prev.loc[common_idx]) / ma_prev.loc[common_idx]
        spikes = inc[inc >= CFG["trend_increase_pct"]]
        for ts, pct in spikes.tail(5).items():
            alertas.append({
                "id": _new_id(),
                "tipo": "tendencia_alcista",
                "variable": var.replace("_kwh","").replace("_litros","").replace("_total_usd",""),
                "unidad": unidad,
                "timestamp": pd.Timestamp(ts).isoformat(),
                "horizonte": "histórico",
                "valor": round(ma_now.loc[ts], 3),
                "umbral": round(ma_prev.loc[ts], 3),
                "porcentaje_sobre_umbral": round(float(pct) * 100, 2),
                "severidad": _severity(float(pct)),
                "mensaje": f"Tendencia alcista de {var} (MA{CFG['trend_window_days']}d) +{float(pct)*100:.1f}%.",
                "recomendacion": "Revisa hábitos recientes y programa recordatorios de ahorro.",
            })

    # Orden final por severidad y fecha
    order = {"crítica": 0, "moderada": 1, "leve": 2}
    alertas.sort(key=lambda a: (order.get(a["severidad"], 3), a["timestamp"]))
    return alertas


def _recomendacion_var(var: str) -> str:
    if var == "energia":
        return "Evita horas pico, apaga standby y optimiza AC/nevera."
    if var == "agua":
        return "Reduce tiempos de ducha y revisa fugas en sanitarios."
    return "Ajusta el uso en horarios más baratos y evita consumos innecesarios."


# =========================
# 6) MAIN
# =========================
def main():
    os.makedirs(PRED_DIR, exist_ok=True)

    print("Cargando dataset base…")
    df_base = cargar_dataset_base(DATASET_CSV)

    print("Cargando predicciones…")
    df_pred = cargar_predicciones(PRED_JSON)

    print("Generando alertas…")
    alertas = generar_alertas(df_base, df_pred)

    out = {
        "generado_en": datetime.now().isoformat(),
        "presupuesto_mensual_usd": PRESUPUESTO_MENSUAL_USD,
        "total_alertas": len(alertas),
        "alertas": alertas[:500]  # evita JSON gigante
    }

    with open(OUT_ALERTAS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"OK → {OUT_ALERTAS_JSON}")
    print(f"Total alertas: {len(alertas)}")


if __name__ == "__main__":
    main()
