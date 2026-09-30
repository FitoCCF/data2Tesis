#!/usr/bin/env python3
# ============================================================
# src/pipeline/unificacion.py — Etapa 1b: tabla maestra del courier (A + B)
# ============================================================
# A (BD, muestreo con leyes) y B (PI, operación continua) son el MISMO
# instrumento: cada muestra de la BD es una lectura que también quedó en el
# PI. Concatenarlas duplicaría lecturas; aquí se une cada muestra de la BD
# con SU lectura del PI y el resultado es una fila por lectura real.
#
# Calce por VALOR, no por hora ni por instance: los 4 metales redondeados a
# entero (la BD los guarda enteros) identifican la lectura. Medido el
# 2026-09-30 sobre las lecturas de la etapa 1a: 122/122 muestras de la BD
# desde 2025-07-13 calzan con exactamente una lectura del PI, y n6sc también
# coincide en las 122. La hora de la BD va una mediana de ~6 min detrás del
# inicio de la lectura en el PI (p95 28 min), y 3 filas de jun-2026 están
# corridas exactamente 12 h (error AM/PM en la BD): para eso la ventana.
#
# Tipos de fila (columna 'fuente'):
#   pi     lectura del PI sin muestra de laboratorio (la gran mayoría)
#   ambos  lectura del PI que coincide con una muestra de la BD: intensidades
#          y hora del PI, leyes y banderas de la BD
#   bd     muestra de la BD sin lectura en el PI (todo lo anterior a 2025-07-13
#          y cualquier muestra posterior que no calce)
#
# Las leyes NO se propagan a lecturas vecinas: solo tienen ley las filas
# 'ambos'/'bd' donde el laboratorio realmente analizó.
#
# Ejecutable de forma independiente:
#   python -m src.pipeline.unificacion
# ============================================================

import pandas as pd

from .config import COLS_BD, VENTANA_CALCE, UMBRAL_HORA_BD

METALES = ["n1fe", "n2cu", "n3zn", "n4mo"]
CANALES = METALES + ["n6sc"]
LEYES = COLS_BD["leyes"]
COLS_BD_EXTRA = ["instance", "instance_duplicada", "leyes_sospechosas", "motivo_sospecha"]
COLS_PI_EXTRA = ["sostenida_s", "sostenida", "n6sc_desfase_s", "n6sc_desfasado"]


def _llave(df: pd.DataFrame) -> pd.Series:
    """Los 4 metales redondeados a entero como texto: identifica la lectura."""
    return df[METALES].round(0).astype("Int64").astype(str).agg("|".join, axis=1)


def calzar(bd: pd.DataFrame, pi: pd.DataFrame, ventana: str = VENTANA_CALCE) -> pd.DataFrame:
    """Para cada fila de la BD, la lectura del PI con los mismos 4 metales más
    cercana en el tiempo dentro de ±ventana. Retorna idx_bd, idx_pi, dt_bd_s
    (hora BD - hora PI) solo de las que calzan."""
    b = pd.DataFrame({"idx_bd": bd.index, "ts_bd": bd["ts"], "k": _llave(bd)})
    p = pd.DataFrame({"idx_pi": pi.index, "ts_pi": pi["ts"], "k": _llave(pi)})
    m = b.merge(p, on="k")
    m["dt_bd_s"] = (m["ts_bd"] - m["ts_pi"]).dt.total_seconds()
    m = m[m["dt_bd_s"].abs() <= pd.Timedelta(ventana).total_seconds()]
    m = m.loc[m["dt_bd_s"].abs().groupby(m["idx_bd"]).idxmin()]         # la más cercana por fila de la BD
    if m["idx_pi"].duplicated().any():
        raise ValueError("dos filas de la BD calzan con la misma lectura del PI: "
                         f"{m.loc[m['idx_pi'].duplicated(keep=False), 'ts_bd'].tolist()}")
    return m[["idx_bd", "idx_pi", "dt_bd_s"]].reset_index(drop=True)


def unificar(bd: pd.DataFrame, pi: pd.DataFrame, ventana: str = VENTANA_CALCE,
             umbral_hora: str = UMBRAL_HORA_BD) -> tuple[pd.DataFrame, dict]:
    """Salidas de la etapa 1a (BD limpia, lecturas del PI) -> tabla maestra."""
    bd = bd.reset_index(drop=True)
    pi = pi.reset_index(drop=True)
    m = calzar(bd, pi, ventana)

    # PI: todas sus lecturas; las que calzan reciben leyes y banderas de la BD
    u_pi = pi[["ts"] + CANALES + COLS_PI_EXTRA].copy()
    u_pi["fuente"] = "pi"
    de_bd = bd.loc[m["idx_bd"], LEYES + COLS_BD_EXTRA + ["ts"]].rename(columns={"ts": "ts_bd"})
    de_bd.index = m["idx_pi"].to_numpy()
    u_pi = u_pi.join(de_bd)
    u_pi.loc[m["idx_pi"], "dt_bd_s"] = m["dt_bd_s"].to_numpy()
    u_pi.loc[m["idx_pi"], "fuente"] = "ambos"

    # BD sin calce: entran con sus propios valores y hora
    sueltas = bd.drop(index=m["idx_bd"])
    u_bd = sueltas[["ts"] + CANALES + LEYES + COLS_BD_EXTRA].copy()
    u_bd["ts_bd"] = u_bd["ts"]
    u_bd["fuente"] = "bd"

    u = pd.concat([u_bd, u_pi], ignore_index=True).sort_values("ts").reset_index(drop=True)
    u["hora_bd_desfasada"] = u["dt_bd_s"].abs() > pd.Timedelta(umbral_hora).total_seconds()
    # banderas booleanas sin NaN: una fila 'bd' no tiene info de PI (False) y una
    # 'pi' no tiene instance (False); así el CSV se relee como bool, no como texto
    for b in ["sostenida", "n6sc_desfasado", "instance_duplicada"]:
        u[b] = u[b].astype("boolean").fillna(False).astype(bool)
    u = u[["ts", "fuente"] + CANALES + LEYES
          + ["ts_bd", "dt_bd_s", "hora_bd_desfasada", "instance", "instance_duplicada",
             "leyes_sospechosas", "motivo_sospecha"] + COLS_PI_EXTRA]

    desde_pi = bd["ts"] >= pi["ts"].min() - pd.Timedelta(ventana)
    r = {"filas": len(u),
         "por_fuente": u["fuente"].value_counts().to_dict(),
         "bd_en_periodo_pi": int(desde_pi.sum()),
         "bd_en_periodo_pi_sin_calce": int((desde_pi & ~bd.index.isin(m["idx_bd"])).sum()),
         "hora_bd_desfasada": int(u["hora_bd_desfasada"].sum()),
         "dt_bd_s_mediana": float(m["dt_bd_s"].median()),
         "filas_con_ley": int(u[LEYES].notna().any(axis=1).sum())}
    return u, r


def _main():
    from .config import LIMPIO_COURIER_BD, LIMPIO_COURIER_PI, UNIFICADO

    bd = pd.read_csv(LIMPIO_COURIER_BD, parse_dates=["ts"])
    pi = pd.read_csv(LIMPIO_COURIER_PI, parse_dates=["ts"])
    u, r = unificar(bd, pi)
    u.to_csv(UNIFICADO, index=False)
    print(f"{LIMPIO_COURIER_BD.name} + {LIMPIO_COURIER_PI.name} -> {UNIFICADO.name}")
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _main()
