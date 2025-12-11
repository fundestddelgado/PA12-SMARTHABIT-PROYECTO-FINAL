import os
import re
import pandas as pd
from pathlib import Path

# ============================
# CONFIGURACIÓN
# ============================
BASE = r"C:\Users\diego\.cache\kagglehub\datasets\programmerrdai\ampds-the-almanac-of-minutely-power-dataset\versions\1"
OUTPUT = r"C:\Users\diego\Desktop\Proyecto SamSung Final\Data Set\dataset_consumo_total_1y.csv"

# Tarifas por defecto
TARIFA_KWH_DEF = 0.16        # USD/kWh (tarifa residencial típica)
TARIFA_LITRO_DEF = 0.0015    # USD/L ≈ 1.5 USD/m³

# Archivos de agua
WATER_FILES = ["Water_DWW.csv", "Water_HTW.csv", "Water_WHW.csv"]


# =====================================================
# Funciones auxiliares
# =====================================================

def detectar_columna(df, preferidas, alternativas=None):
    preferidas = preferidas or []
    alternativas = alternativas or []

    for c in preferidas:
        if c in df.columns:
            return c

    for pat in alternativas:
        for c in df.columns:
            if re.search(pat, str(c), re.IGNORECASE):
                return c

    raise ValueError("No se encontró una columna esperada (tiempo o valor).")


def cargar_tarifas_si_hay(base_dir):
    tarifa_kwh = TARIFA_KWH_DEF
    tarifa_litro = TARIFA_LITRO_DEF

    # -------------------------------------------
    # ELECTRICIDAD
    # -------------------------------------------
    elec_billing = Path(base_dir, "Electricity_Billing.csv")
    if elec_billing.exists():
        try:
            df = pd.read_csv(elec_billing)
            cols = [c for c in df.columns if re.search("rate|tarif|price", c, re.I)]
            if cols:
                v = pd.to_numeric(df[cols[0]], errors="coerce").dropna()
                if len(v):
                    t = float(v.mean())

                    # Rango razonable de tarifa
                    if 0.05 <= t <= 1.0:
                        tarifa_kwh = t
                        print(f"[INFO] Tarifa electricidad detectada: {tarifa_kwh:.4f} USD/kWh")
                    else:
                        print(f"[WARN] Tarifa eléctrica fuera de rango ({t:.4f}). Se usa valor por defecto.")
        except Exception as e:
            print(f"[WARN] No se pudo leer tarifa eléctrica: {e}")

    # -------------------------------------------
    # AGUA
    # -------------------------------------------
    water_billing = Path(base_dir, "Water_Billing.csv")
    if water_billing.exists():
        try:
            df = pd.read_csv(water_billing)
            cols = [c for c in df.columns if re.search("rate|tarif|price", c, re.I)]
            if cols:
                v = pd.to_numeric(df[cols[0]], errors="coerce").dropna()
                if len(v):
                    t_m3 = float(v.mean())
                    t_l = t_m3 / 1000.0

                    if 0.0002 <= t_l <= 0.01:
                        tarifa_litro = t_l
                        print(f"[INFO] Tarifa agua detectada: {tarifa_litro:.6f} USD/L")
                    else:
                        print(f"[WARN] Tarifa agua fuera de rango ({t_l:.6f}). Se usa valor por defecto.")
        except Exception as e:
            print(f"[WARN] No se pudo leer tarifa de agua: {e}")

    return tarifa_kwh, tarifa_litro


# =====================================================
# ELECTRICIDAD
# =====================================================

def construir_electricidad_por_hora(base_dir):
    archivos = sorted([
        p for p in Path(base_dir).glob("Electricity_*.csv")
        if not re.search("Billing|Monthly|Statements", p.name, re.I)
    ])

    if not archivos:
        raise SystemExit("[ERROR] No existen archivos de electricidad.")

    total_h = None
    print(f"[INFO] {len(archivos)} archivos de electricidad encontrados.")

    for p in archivos:
        try:
            df = pd.read_csv(p)

            col_time = detectar_columna(df,
                ["unix_ts", "unixTS", "timestamp", "time"],
                alternativas=[r"unix", r"time"]
            )

            col_P = detectar_columna(df,
                ["P"],
                alternativas=[r"^P$", r"power", r"P_W", r"Watts?"]
            )

            # Convertir a datetime
            if "unix" in col_time.lower():
                ts = pd.to_datetime(df[col_time], unit="s", errors="coerce")
            else:
                ts = pd.to_datetime(df[col_time], errors="coerce")

            s = pd.Series(pd.to_numeric(df[col_P], errors="coerce").fillna(0.0).values,
                          index=ts).sort_index()

            # Resample a minuto y hora
            s_1min = s.resample("min").mean()
            kwh_h = s_1min.resample("h").sum() / 60000.0  # W/min → kWh

            total_h = kwh_h if total_h is None else total_h.add(kwh_h, fill_value=0.0)

        except Exception as e:
            print(f"[WARN] Saltando {p.name}: {e}")

    if total_h is None:
        raise SystemExit("[ERROR] No se pudo generar energía por hora.")

    return total_h.rename("energia_kwh")


# =====================================================
# AGUA
# =====================================================

def construir_agua_por_hora(base_dir):
    series = []

    for fname in WATER_FILES:
        p = Path(base_dir, fname)
        if not p.exists():
            print(f"[WARN] Falta archivo de agua {fname}")
            continue

        try:
            df = pd.read_csv(p)

            col_time = detectar_columna(df,
                ["unix_ts", "unixTS", "timestamp", "time"],
                alternativas=[r"unix", r"time"]
            )
            col_flow = detectar_columna(df,
                ["avg_rate", "flow"],
                alternativas=[r"rate", r"flow"]
            )

            # Convertir timestamps
            if "unix" in col_time.lower():
                ts = pd.to_datetime(df[col_time], unit="s", errors="coerce")
            else:
                ts = pd.to_datetime(df[col_time], errors="coerce")

            s = pd.Series(pd.to_numeric(df[col_flow], errors="coerce").fillna(0.0).values,
                          index=ts).sort_index()

            s_1min = s.resample("min").mean()
            l_h = s_1min.resample("h").sum()

            series.append(l_h)

        except Exception as e:
            print(f"[WARN] Error leyendo {fname}: {e}")

    if not series:
        raise SystemExit("[ERROR] No hay datos válidos de agua.")

    total = series[0]
    for s in series[1:]:
        total = total.add(s, fill_value=0.0)

    return total.rename("agua_litros")


# =====================================================
# MAIN
# =====================================================

def main():

    print("[STEP] Procesando electricidad…")
    energia_h = construir_electricidad_por_hora(BASE)

    print("[STEP] Procesando agua…")
    agua_h = construir_agua_por_hora(BASE)

    print("[STEP] Uniendo series…")
    df = pd.concat([energia_h, agua_h], axis=1)

    # Variables temporales
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Tarifas
    tarifa_kwh, tarifa_litro = cargar_tarifas_si_hay(BASE)

    df["costo_luz_usd"] = df["energia_kwh"].fillna(0) * tarifa_kwh
    df["costo_agua_usd"] = df["agua_litros"].fillna(0) * tarifa_litro
    df["costo_total_usd"] = df["costo_luz_usd"] + df["costo_agua_usd"]

    # Reset index (timestamp) y renombrar robustamente
    df = df.reset_index()

    # Detectar la columna datetime
    time_col = None
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            time_col = c
            break

    if time_col is None:
        time_col = "index" if "index" in df.columns else df.columns[0]

    if time_col != "timestamp":
        df = df.rename(columns={time_col: "timestamp"})

    # Guardar
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"\n[OK] Dataset generado en:\n{OUTPUT}")
    print(f"[INFO] Filas: {len(df):,}")
    print(f"[INFO] Rango temporal: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print("\nPrimeras filas:")
    print(df.head())


if __name__ == "__main__":
    main()
