#!/usr/bin/env python3
# ============================================================
# CELDA 14 — Repite el experimento 13 pero a la granularidad de la cabeza,
# no del courier: elimina el artefacto de bloques repetidos.
# ============================================================
# El experimento 13 alineó la cabeza (baja frecuencia, ~72h) al stream del
# courier (alta frecuencia, ~15min) por último valor conocido: cada lectura
# de cabeza terminaba repetida ~86 veces seguidas, y ESO -no la información
# real- fue lo que movió el BIC (el control barajado colapsaba igual).
#
# Aquí se hace al revés: se agrega el courier a la granularidad de la cabeza.
# Cada lectura de cabeza L2 define una ÉPOCA = [cabeza_i, cabeza_i+1); se
# promedian las features del courier dentro de esa época. Resultado: ~266
# filas independientes, sin repetición artificial. Si la cabeza real ayuda
# aquí, ya no puede ser por el artefacto de bloques.
#
# Costo: n cae de 23,058 a ~266. Es la granularidad real de la información
# nueva -- no hay forma de evitarlo sin inventar dato que no existe.
#
# Uso:
#   pixi run python notebooks/14_experimento_cluster_cabeza_epocas.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import psycopg2

from src.pipeline import config as CFG
from src.pipeline.clustering import diagnostico_k

SAMPLE_ID_CABEZA_L2 = 18  # "Rebose Hidrociclones L2" en works4cdp_sample


def cargar_cabeza_l2() -> pd.DataFrame:
    """Idéntico al de la celda 13: intensidades crudas de la cabeza L2 desde
    la BD, + fracciones de cierre. Copiado en vez de importado porque el
    nombre del módulo 13 empieza con dígito (no es un import válido)."""
    conn = psycopg2.connect(host="localhost", port=5432, user="myuser",
                            password="mypassword", dbname="mydb")
    q = f"""
        SELECT timestamp AS ts, n1fe, n2cu, n3zn, n4mo
        FROM works4cdp_assay
        WHERE sample_id = {SAMPLE_ID_CABEZA_L2}
          AND n1fe IS NOT NULL AND n2cu IS NOT NULL
          AND n3zn IS NOT NULL AND n4mo IS NOT NULL
        ORDER BY timestamp
    """
    cab = pd.read_sql(q, conn, parse_dates=["ts"])
    conn.close()
    cab["ts"] = cab["ts"].dt.tz_localize(None)

    suma = cab[["n1fe", "n2cu", "n3zn", "n4mo"]].sum(axis=1).replace(0, np.nan)
    for m in ["n1fe", "n2cu", "n3zn", "n4mo"]:
        cab[f"cab_{m}_f"] = cab[m] / suma
    return cab.dropna(subset=[f"cab_{m}_f" for m in ["n1fe", "n2cu", "n3zn", "n4mo"]])


def construir_epocas(df: pd.DataFrame, cab: pd.DataFrame, cols_cab: list[str]) -> pd.DataFrame:
    """Una fila por época = [cabeza_i, cabeza_i+1). Promedia FEATS_CLUSTER del
    courier dentro de la época y la empareja con la cabeza que la abrió."""
    cab = cab.sort_values("ts").reset_index(drop=True)
    filas = []
    for i in range(len(cab) - 1):
        t0, t1 = cab.loc[i, "ts"], cab.loc[i + 1, "ts"]
        bloque = df[(df["ts"] >= t0) & (df["ts"] < t1)]
        if bloque.empty:
            continue
        fila = {"ts": t0, "n_courier": len(bloque)}
        fila.update(bloque[CFG.FEATS_CLUSTER].mean().to_dict())
        fila.update({c: cab.loc[i, c] for c in cols_cab})
        filas.append(fila)
    return pd.DataFrame(filas)


def _main():
    print("=" * 62)
    print("PASO 1 — Construir épocas (una fila por lectura de cabeza L2)")
    print("=" * 62)
    df = pd.read_csv(CFG.DATA_PROCESSED / "intensidad_cobre_24_scaled.csv")
    df["ts"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    df = df.sort_values("ts").reset_index(drop=True)

    cab = cargar_cabeza_l2()
    cols_cab = ["cab_n1fe_f", "cab_n2cu_f", "cab_n3zn_f", "cab_n4mo_f"]

    E = construir_epocas(df, cab, cols_cab)
    print(f"Épocas construidas: {len(E)}  (de {len(cab)} lecturas de cabeza)")
    print(f"Cobertura de courier por época: mediana={E['n_courier'].median():.0f} "
          f"filas, min={E['n_courier'].min()}, max={E['n_courier'].max()}")
    E = E[E["n_courier"] >= 5].reset_index(drop=True)   # épocas con muy poco courier no son confiables
    print(f"Épocas con >= 5 lecturas de courier: {len(E)}")

    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 62)
    print("PASO 2 — BASELINE a nivel de época (solo Concentrado Colectivo)")
    print("=" * 62)
    X_base = StandardScaler().fit_transform(E[CFG.FEATS_CLUSTER].values)
    diagnostico_k(X_base, k_max=6)   # con ~260 filas, k>6 tiene poco sentido

    print("\n" + "=" * 62)
    print("PASO 3 — AUMENTADO a nivel de época (+ cabeza L2 real)")
    print("=" * 62)
    X_aum = StandardScaler().fit_transform(E[CFG.FEATS_CLUSTER + cols_cab].values)
    diagnostico_k(X_aum, k_max=6)

    print("\n" + "=" * 62)
    print("PASO 4 — Control negativo a nivel de época (cabeza barajada)")
    print("=" * 62)
    rng = np.random.default_rng(CFG.RANDOM_STATE)
    E_ctrl = E.copy()
    orden = rng.permutation(len(E_ctrl))
    E_ctrl[cols_cab] = E_ctrl[cols_cab].values[orden]
    X_ctrl = StandardScaler().fit_transform(E_ctrl[CFG.FEATS_CLUSTER + cols_cab].values)
    diagnostico_k(X_ctrl, k_max=6)


if __name__ == "__main__":
    _main()
