#!/usr/bin/env python3
# ============================================================
# src/pipeline/limpieza_fuentes.py — Etapa 1a: limpieza por fuente
# ============================================================
# Cada crudo de la etapa 0 tiene problemas propios que no se pueden tratar
# con una limpieza común (ya pasó una vez: el filtro de "sensor congelado"
# aplicado al PI borraba miles de lecturas válidas). Aquí solo van las reglas
# DETERMINISTAS de cada fuente. El IsolationForest y el filtro de congelado
# van en la etapa 1c, sobre la tabla unificada.
#
# Regla de fondo: una fila con una medición real no se borra por un campo
# malo. Se borra solo lo que no es una medición (analizador detenido, código
# de error en todos los canales, republicación del mismo valor); lo dudoso se
# MARCA con una bandera y la decisión queda para después.
#
# Criterios medidos el 2026-09-30 sobre los crudos hasta 2026-08-31
# (docs/bitacora_analisis.md):
#   PI  - El courier entrega una lectura del concentrado cada ~21 min. El PI
#         la vuelve a archivar cada 100 s (ExcMax=100 en los 5 tags) hasta la
#         siguiente: ~14 eventos por lectura real.
#       - Los 4 metales cambian siempre juntos. n6sc llega en el mismo segundo
#         salvo en 2026-03..07, cuando la interfaz lo archivó 1-60 s aparte
#         (filas partidas, pico de 6,399 en mayo). Coincide con el período de
#         las 18 muestras de la BD que no aparecen en el PI.
#   BD  - n5ech5/n7ech7 vacíos para sample_id=24.
#       - 4 instance duplicadas: la misma lectura registrada dos veces, una
#         con las leyes (hora redondeada a hh:mm:00) y otra sin ellas.
#       - Ley en 0 = el laboratorio no analizó ese elemento (pZn 72, pIns 16,
#         pSol 18). Nunca se borra la fila por eso.
#
# Ejecutable de forma independiente:
#   python -m src.pipeline.limpieza_fuentes
# ============================================================

import numpy as np
import pandas as pd

from .config import (COLS_BD, TAGS_COURIER, TOL_EVENTO, TOL_N6SC, UMBRAL_SOSTENIDA,
                     Z_LEY_SOSPECHOSA, Z_CANAL_RESPALDO, CANAL_DE_LEY)

CANALES_PI = list(TAGS_COURIER.values())                  # n1fe, n2cu, n3zn, n4mo, n6sc
METALES_PI = [c for c in CANALES_PI if c != "n6sc"]
LEYES_BD = COLS_BD["leyes"]
LEYES_COMPOSITO = ["fe", "cu", "ins", "mo"]


# ============================================================
# PI — fuente B
# ============================================================
def _inicio_de_corrida(x: pd.DataFrame) -> pd.Series:
    """True en la fila donde empieza un valor nuevo (alguna columna distinta de
    la fila anterior). NaN se compara como valor: NaN->NaN no es cambio."""
    prev = x.shift()
    distinto = (x != prev) & ~(x.isna() & prev.isna())
    return distinto.any(axis=1)


def limpiar_pi(df: pd.DataFrame, tol_evento: str = TOL_EVENTO, tol_n6sc: str = TOL_N6SC,
               umbral_sostenida: str = UMBRAL_SOSTENIDA) -> tuple[pd.DataFrame, dict]:
    """Eventos archivados del PI -> una fila por lectura real del courier.

    1. -9999 y 0 pasan a NaN por canal. Una fila que queda sin ningún canal
       no era una medición (todo -9999 = error; todo 0 = analizador detenido).
    2. La lectura la definen los 4 metales. Primero se juntan los eventos de
       metales a <= tol_evento entre sí (a veces un metal se archiva 1 s antes
       que los otros y partiría la lectura en dos filas incompletas). Una
       lectura nueva es un cambio en cualquiera de ellos. Las republicaciones se colapsan a su primer
       timestamp; 'sostenida_s' es cuánto siguió republicándose.
    3. n6sc se pega desde su propio evento más cercano dentro de ±tol_n6sc,
       prefiriendo el instante en que n6sc cambió (así no se toma un valor
       viejo republicado). 'n6sc_desfase_s' = segundos entre ambos; 0 en los
       meses normales.

    Retorna (lecturas, resumen de cuántas filas tocó cada regla).
    """
    df = df.sort_values("ts").reset_index(drop=True)
    x = df[CANALES_PI]
    r = {"eventos_crudos": len(df),
         "eventos_todo_-9999": int((x == -9999).all(axis=1).sum()),
         "eventos_todo_0": int((x == 0).all(axis=1).sum())}

    x = x.mask((x == -9999) | (x == 0))                   # 1. código de error y ceros -> NaN por canal
    r["canales_a_nan"] = int(x.isna().sum().sum() - df[CANALES_PI].isna().sum().sum())
    df = df.assign(**{c: x[c] for c in CANALES_PI})
    df = df[df[CANALES_PI].notna().any(axis=1)]
    r["eventos_sin_medicion"] = r["eventos_crudos"] - len(df)

    # 2. lecturas: eventos con metales, colapsados por cambio de valor
    met = df[df[METALES_PI].notna().any(axis=1)][["ts"] + METALES_PI].reset_index(drop=True)
    ciclo = (met["ts"].diff() > pd.Timedelta(tol_evento)).cumsum()
    met = met.groupby(ciclo).agg(ts=("ts", "first"), **{c: (c, "first") for c in METALES_PI})
    r["eventos_metales_unidos"] = int(len(ciclo) - len(met))
    # Cada canal se compara contra su ÚLTIMO valor conocido (el estado que el PI
    # sostiene), no contra la fila anterior: una republicación que llega
    # repartida en varios eventos no es una lectura nueva.
    estado = met[METALES_PI].ffill()
    corrida = _inicio_de_corrida(estado).cumsum()
    lect = (estado.assign(ts=met["ts"]).groupby(corrida)
            .agg(ts=("ts", "first"), ts_ultimo=("ts", "last"), **{c: (c, "first") for c in METALES_PI}))
    lect["sostenida_s"] = (lect.pop("ts_ultimo") - lect["ts"]).dt.total_seconds()
    r["eventos_republicados"] = len(met) - len(lect)

    # 3. n6sc: primero su instante de cambio más cercano; si no hay, cualquier evento
    sc = df[df["n6sc"].notna()][["ts", "n6sc"]].reset_index(drop=True)
    sc_cambio = sc[_inicio_de_corrida(sc[["n6sc"]])]
    tol = pd.Timedelta(tol_n6sc)
    pegar = lambda fuente: pd.merge_asof(
        lect[["ts"]], fuente.rename(columns={"ts": "ts_sc"}), left_on="ts", right_on="ts_sc",
        direction="nearest", tolerance=tol)
    a, b = pegar(sc_cambio), pegar(sc)
    usar_b = a["n6sc"].isna()
    lect["n6sc"] = a["n6sc"].where(~usar_b, b["n6sc"]).to_numpy()
    ts_sc = a["ts_sc"].where(~usar_b, b["ts_sc"])
    lect["n6sc_desfase_s"] = (ts_sc - a["ts"]).dt.total_seconds().to_numpy()

    lect["n6sc_desfasado"] = lect["n6sc_desfase_s"].fillna(0).ne(0)
    lect["sostenida"] = lect["sostenida_s"] > pd.Timedelta(umbral_sostenida).total_seconds()
    lect = lect[["ts"] + CANALES_PI + ["sostenida_s", "sostenida", "n6sc_desfase_s", "n6sc_desfasado"]]
    lect = lect.reset_index(drop=True)

    r.update({"lecturas": len(lect),
              "lecturas_sin_n6sc": int(lect["n6sc"].isna().sum()),
              "lecturas_n6sc_desfasado": int(lect["n6sc_desfasado"].sum()),
              "lecturas_sostenidas": int(lect["sostenida"].sum()),
              "lecturas_metales_incompletos": int(lect[METALES_PI].isna().any(axis=1).sum())})
    return lect, r


# ============================================================
# BD — fuente A
# ============================================================
def _z_robusto(s: pd.Series) -> pd.Series:
    """(s - mediana) / (1.4826·MAD). NaN si la MAD es 0."""
    mad = 1.4826 * (s - s.median()).abs().median()
    return (s - s.median()) / mad if mad > 0 else s * np.nan


def _marcar_leyes_sospechosas(df: pd.DataFrame, leyes: list[str], z: float,
                              z_canal: float) -> tuple[pd.Series, pd.Series]:
    """Marca para revisión (no filtro) las leyes que el laboratorio reportó
    extremas y el courier no vio.

    Una ley es sospechosa si está a más de z MAD robustas de su mediana Y su
    canal del courier (CANAL_DE_LEY) NO se desvía más de z_canal en la misma
    dirección. Si el canal también se desvía, el material venía así (medido:
    los pZn/pMo altos reales traen n3zn/n4mo altos). pIns/pSol no tienen canal
    propio: basta con que la ley sea extrema.

    Retorna (leyes marcadas separadas por ';', motivo legible).
    """
    marcas = pd.Series("", index=df.index)
    motivos = pd.Series("", index=df.index)
    for ley in leyes:
        zl = _z_robusto(df[ley])
        extrema = zl.abs() > z
        canal = CANAL_DE_LEY.get(ley)
        if canal:
            zc = _z_robusto(df[canal])
            respaldada = (zc.abs() > z_canal) & (np.sign(zc) == np.sign(zl))
            sospecha = extrema & ~respaldada
            desc = lambda i: (f"{ley} {'alta' if zl[i] > 0 else 'baja'} (z={zl[i]:.1f}), "
                              f"{canal} z={zc[i]:.1f}")
        else:
            sospecha = extrema
            desc = lambda i: f"{ley} {'alta' if zl[i] > 0 else 'baja'} (z={zl[i]:.1f}), sin canal"
        for i in df.index[sospecha.fillna(False)]:
            marcas[i] += ley + ";"
            motivos[i] += desc(i) + "; "
    return marcas.str.rstrip(";"), motivos.str.rstrip("; ")


def limpiar_bd(df: pd.DataFrame, z_sospecha: float = Z_LEY_SOSPECHOSA,
               z_canal: float = Z_CANAL_RESPALDO) -> tuple[pd.DataFrame, dict]:
    """Muestreo con leyes (BD) -> una fila por lectura, con ts y leyes en NaN
    donde el laboratorio no analizó.

    1. Canales de intensidad 100% vacíos se quitan (n5ech5, n7ech7 en sample 24).
    2. ts = date + time (hora local, sin fracción de segundo).
    3. instance duplicada = misma lectura registrada dos veces: se une en una
       fila con las intensidades (idénticas) y las leyes de la que las traiga.
       Queda la hora de la fila sin leyes, que es la del analizador.
    4. Ley en 0 -> NaN (no analizada). No se borra la fila.
    5. Ley extrema que su canal del courier no respalda -> columnas
       'leyes_sospechosas' y 'motivo_sospecha' (ver _marcar_leyes_sospechosas).
    """
    intens = [c for c in COLS_BD["intensidades"] if c in df.columns]
    vacios = [c for c in intens if df[c].isna().all()]
    df = df.drop(columns=vacios)
    intens = [c for c in intens if c not in vacios]
    r = {"filas_crudas": len(df), "canales_vacios_quitados": vacios}

    hora = df["time"].astype(str).str.split(".").str[0]
    df.insert(0, "ts", pd.to_datetime(df["date"].astype(str) + " " + hora))
    df = df.drop(columns=["date", "time"])

    # 3. duplicados de instance: la fila sin leyes primero -> su ts y sus intensidades
    tiene_ley = df[LEYES_BD].notna().any(axis=1)
    dup = df["instance"].duplicated(keep=False)
    difieren = df[dup].groupby("instance")[intens].nunique().gt(1).any(axis=1)
    if difieren.any():
        raise ValueError(f"instance duplicada con intensidades distintas: {list(difieren[difieren].index)}")
    df = (df.assign(_ley=tiene_ley).sort_values(["instance", "_ley"])
            .groupby("instance", as_index=False, sort=False).first()   # first() salta NaN -> toma las leyes de la otra
            .drop(columns="_ley"))
    df["instance_duplicada"] = df["instance"].isin(difieren.index)  # instance que venían duplicadas
    r["instance_unidas"] = len(difieren)
    df = df.sort_values("ts").reset_index(drop=True)

    # 4. ley en 0 = no analizada
    ceros = (df[LEYES_BD] == 0)
    r["leyes_0_a_nan"] = {k: int(v) for k, v in ceros.sum().items() if v}
    df[LEYES_BD] = df[LEYES_BD].mask(ceros)

    # 5. marca de sospecha
    df["leyes_sospechosas"], df["motivo_sospecha"] = _marcar_leyes_sospechosas(df, LEYES_BD, z_sospecha, z_canal)
    r.update({"filas": len(df),
              "filas_con_ley": int(df[LEYES_BD].notna().any(axis=1).sum()),
              "filas_ley_sospechosa": int(df["leyes_sospechosas"].ne("").sum())})
    return df, r


# ============================================================
# Compósito — fuente C
# ============================================================
def limpiar_composito(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compósito de 12 h: orden por tiempo, un ensayo por instante, ley en 0 -> NaN."""
    df = df.sort_values("ts")
    r = {"filas_crudas": len(df), "ts_duplicado": int(df["ts"].duplicated().sum())}
    df = df.drop_duplicates("ts", keep="last").reset_index(drop=True)
    leyes = [c for c in LEYES_COMPOSITO if c in df.columns]
    ceros = df[leyes] == 0
    r["leyes_0_a_nan"] = {k: int(v) for k, v in ceros.sum().items() if v}
    df[leyes] = df[leyes].mask(ceros)
    r["filas"] = len(df)
    return df, r


# ============================================================
# CLI
# ============================================================
def _main():
    import argparse
    from .config import (RAW_COURIER_PI, RAW_COURIER_BD, RAW_COMPOSITO,
                         LIMPIO_COURIER_PI, LIMPIO_COURIER_BD, LIMPIO_COMPOSITO)

    ap = argparse.ArgumentParser(description="Etapa 1a: limpieza por fuente")
    ap.add_argument("--fuentes", nargs="+", choices=["bd", "pi", "composito"], default=["bd", "pi", "composito"])
    args = ap.parse_args()

    tareas = {"bd": (limpiar_bd, RAW_COURIER_BD, LIMPIO_COURIER_BD),
              "pi": (limpiar_pi, RAW_COURIER_PI, LIMPIO_COURIER_PI),
              "composito": (limpiar_composito, RAW_COMPOSITO, LIMPIO_COMPOSITO)}
    for nombre in args.fuentes:
        fn, entrada, salida = tareas[nombre]
        df, r = fn(pd.read_csv(entrada, parse_dates=["ts"] if nombre != "bd" else None))
        df.to_csv(salida, index=False)
        print(f"[{nombre}] {entrada.name} -> {salida.name}")
        for k, v in r.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    _main()
