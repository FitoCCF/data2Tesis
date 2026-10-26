#!/usr/bin/env python3
# ============================================================
# scripts/en_linea_cabeza.py — estimación EN LÍNEA de Cu, Fe y Mo de CABEZA (rebose HC L2)
# ============================================================
# Tercer monitor (rama AlimConc, bitácora §23.2). Independiente de en_linea.py:
# lee y escribe solo sus propios archivos.
#
# Al arrancar ajusta el modelo UNA vez: Ridge global por ley sobre la media de
# las features en la ventana centrada de cada compósito de cabeza de train
# (2026-03-01 -> FIN_TRAIN_CABEZA). Train no cambia con datos nuevos, así que
# no hay reentrenamiento en el bucle. Cada ciclo:
#   1. extrae del PI lo nuevo de RebHidroL2 y del compósito de cabeza
#      (7100AIP011-014MAN), incremental y con superposición;
#   2. estima cada lectura (lineal: la media de las estimaciones es la
#      estimación de la media) y promedia en la ventana centrada de cada
#      compósito, publicado o pendiente;
#   3. Kalman "escala balde" con q elegido en train (causal: solo compósitos
#      ya publicados).
#
# Salidas: data/final/en_linea_cabeza.csv, reports/tendencia_cabeza.{png,html}
#
#   PI_GATEWAY_HOST=<ip> PI_GATEWAY_PORT=<puerto> pixi run python -u scripts/en_linea_cabeza.py --cada 20
#   pixi run python scripts/en_linea_cabeza.py --sin-extraer        # sin PI (desarrollo)
#
# La IP de la pasarela va solo por entorno, nunca en un archivo.
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.pipeline.config import (DATA_FINAL, REPORTS_DIR, RAW_CABEZA, TAGS_CABEZA, BLOQUE_RESAMPLE,
                                 VENTANA_COMPOSITO_H)
from src.evaluacion.evaluar import kalman_serie, Q_GRID, metricas
from scripts.cabeza_intensidades import CANALES, FEATS, MEDIA, MIN_LECTURAS, lecturas, raw_linea
from scripts.cabeza_ley_en_linea import tabla_compositos, _ridge, LEYES, FIN_TRAIN_CABEZA
from scripts.en_linea import MonitorEnLinea, SOLAPE
from scripts import tendencia_en_linea as tend

LINEA = "L2"
SALIDA = DATA_FINAL / "en_linea_cabeza.csv"
DESDE_SALIDA = pd.Timestamp(FIN_TRAIN_CABEZA) + pd.Timedelta(days=1)   # se muestra desde el inicio de test


class MonitorCabeza:
    def __init__(self, extraer: bool = True):
        self.extraer = extraer
        A, _ = tabla_compositos(LINEA)
        tr = A["periodo"] == "train"
        self.modelos = {l: _ridge().fit(A.loc[tr, FEATS], A.loc[tr, l]) for l in LEYES}
        print(f"Modelo de cabeza {LINEA}: Ridge global ajustado con {tr.sum()} compósitos de train "
              f"(hasta {FIN_TRAIN_CABEZA})")

    def adquirir(self) -> None:
        from src.acquisition.from_pi import cliente, _a_ancho, extraer_stream
        unir = MonitorEnLinea._unir
        ruta = raw_linea(LINEA)
        tags = {f"_296290_RebHidro{LINEA}_Canal{k}_ABB": v for k, v in CANALES.items()}
        desde = pd.Timestamp(pd.read_csv(ruta, usecols=["ts"])["ts"].max()) - SOLAPE
        nuevo = _a_ancho(cliente().recorded(list(tags), desde.strftime("%Y-%m-%d %H:%M:%S"),
                                            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), chunk_dias=7), tags)
        n, hasta = unir(None, ruta, nuevo)
        print(f"  {ruta.name}: +{n} filas, hasta {hasta}")
        desde = pd.Timestamp(pd.read_csv(RAW_CABEZA, usecols=["ts"])["ts"].max()) - SOLAPE
        n, hasta = unir(None, RAW_CABEZA, extraer_stream(TAGS_CABEZA, desde.strftime("%Y-%m-%d %H:%M:%S")))
        print(f"  {RAW_CABEZA.name}: +{n} filas, hasta {hasta}")

    def estimar(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        lect = lecturas(LINEA)
        for l in LEYES:
            lect[f"{l}_est"] = self.modelos[l].predict(lect[FEATS])
        cols = [f"{l}_est" for l in LEYES]
        comp = pd.read_csv(RAW_CABEZA, parse_dates=["ts"]).set_index("ts")
        comp = comp[comp.index >= lect.index.min() + MEDIA]

        marcas = list(comp.index)                       # publicados + pendientes hasta el turno en curso
        m, ahora = comp.index.max() + pd.Timedelta(hours=12), pd.Timestamp.now()
        while m - MEDIA <= ahora:
            marcas.append(m)
            m += pd.Timedelta(hours=12)
        filas = []
        for t in marcas:
            w = lect.loc[t - MEDIA: t + MEDIA, cols]
            fila = {"ts": t, "turno": "dia" if t.hour < 12 else "noche", "n_lecturas": len(w),
                    "n_bloques": int(w.resample(BLOQUE_RESAMPLE).size().gt(0).sum()) if len(w) else 0}
            fila.update(w.mean().to_dict() if len(w) >= MIN_LECTURAS else {c: np.nan for c in cols})
            fila.update({l: comp.at[t, l] if t in comp.index else np.nan for l in LEYES})
            filas.append(fila)
        A = pd.DataFrame(filas).sort_values("ts").reset_index(drop=True)
        A["periodo"] = np.where(A["ts"] < DESDE_SALIDA, "train", "test")
        for l in LEYES:                                  # q elegido SOLO con train
            # sin estimación (parada, courier caído) el Kalman no se actualiza: un NaN
            # en la predicción le dejaría el sesgo en NaN para siempre
            B = A[A[f"{l}_est"].notna()]
            tr = (B["periodo"] == "train") & B[l].notna()
            r2q = {q: metricas(B.loc[tr, l], kalman_serie(B, l, q)[0][tr]).get("R2", -np.inf) for q in Q_GRID}
            A[f"{l}_kal"] = kalman_serie(B, l, max(r2q, key=r2q.get))[0].reindex(A.index)
        ultima = lect.index.max()
        A["ventana_completa"] = A["ts"] + MEDIA <= ahora
        A["horas_sin_lectura"] = (ahora - ultima) / pd.Timedelta(hours=1)
        A["publicado"] = A["cu"].notna()
        A["ultima_lectura"] = ultima
        A = A[A["ts"] >= DESDE_SALIDA].reset_index(drop=True)
        A.to_csv(SALIDA, index=False)
        return A, lect

    def ciclo(self) -> None:
        t0 = time.time()
        print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] ciclo en línea (cabeza {LINEA})")
        if self.extraer:
            self.adquirir()
        A, lect = self.estimar()
        h = A["horas_sin_lectura"].iloc[0]
        print(f"  listo en {time.time() - t0:.0f} s; última lectura del courier de cabeza: {A['ultima_lectura'].iloc[0]}"
              + (f"  ** SIN LECTURAS NUEVAS HACE {h:.1f} h **" if h > 1 else ""))
        mostrar(A)
        try:
            cols = [f"{l}_est" for l in LEYES]
            q = lect[cols].quantile([0.005, 0.995])           # solo para el dibujo: fuera las lecturas
            bloques = (lect[cols].where((lect[cols] >= q.iloc[0]) & (lect[cols] <= q.iloc[1]))   # de arranque
                       .resample(BLOQUE_RESAMPLE).mean())
            perfil = tend.Perfil(f"Cabeza (rebose HC {LINEA})", SALIDA, REPORTS_DIR / "tendencia_cabeza.png",
                                 REPORTS_DIR / "tendencia_cabeza.html", [("cu", "Cu", 3), ("fe", "Fe", 2), ("mo", "Mo", 4)],
                                 lambda desde: bloques[bloques.index >= desde - MEDIA].reset_index(),
                                 monitor="en_linea_cabeza.py")
            tend.generar(7, perfil)
            print("  tendencia: reports/tendencia_cabeza.html")
        except Exception as e:
            print(f"  (tendencia no generada: {type(e).__name__}: {str(e)[:200]})")


def mostrar(A: pd.DataFrame, n: int = 6) -> None:
    filas = []
    for f in A.tail(n).itertuples(index=False):
        fila = {"compósito": f.ts.strftime("%m-%d %H:%M"), "lecturas": f.n_lecturas,
                "estado": "lab" if f.publicado else ("ventana cerrada" if f.ventana_completa else "en curso")}
        for l, el, dec in [("cu", "Cu", 3), ("fe", "Fe", 2), ("mo", "Mo", 4)]:
            est, lab = getattr(f, f"{l}_kal"), getattr(f, l)
            fila[el] = ("sin datos" if np.isnan(est) else f"{est:.{dec}f}" if np.isnan(lab)
                        else f"{est:.{dec}f} vs {lab:.{dec}f} ({est - lab:+.{dec}f})")
        filas.append(fila)
    print(pd.DataFrame(filas).to_string(index=False))


def _main():
    ap = argparse.ArgumentParser(description="Estimación en línea de Cu/Fe/Mo de cabeza vs compósito de cabeza")
    ap.add_argument("--cada", type=float, default=0, help="minutos entre ciclos (0 = un solo ciclo)")
    ap.add_argument("--sin-extraer", action="store_true", help="no consulta el PI (modo desarrollo)")
    args = ap.parse_args()
    monitor = MonitorCabeza(extraer=not args.sin_extraer)
    while True:
        try:
            monitor.ciclo()
        except Exception as e:                           # un ciclo caído no tumba el monitor
            print(f"  CICLO FALLÓ: {type(e).__name__}: {str(e)[:500]}")
        if not args.cada:
            break
        time.sleep(args.cada * 60)


if __name__ == "__main__":
    _main()
