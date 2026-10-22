#!/usr/bin/env python3
# ============================================================
# scripts/en_linea.py — estimación EN LÍNEA de Cu, Fe y Mo del Concentrado Colectivo
# ============================================================
# Cada ciclo:
#   1. extrae del PI lo nuevo del courier y del compósito (incremental, con
#      superposición) y lo une a los crudos sin duplicar;
#   2. corre las etapas 1a -> 6 con el bundle de la etapa 5 CONGELADO (no se
#      reentrena nada: todo lo que se ajusta usa periodo == 'train', que no
#      cambia con datos nuevos);
#   3. promedia la estimación sobre la ventana centrada de cada compósito
#      pendiente (07:30 y 19:30) y la lleva a escala balde con el Kalman, cuyo
#      sesgo usa solo compósitos ya publicados (causal);
#   4. cuando el laboratorio publica el compósito, lo compara.
#
# Salida: data/final/en_linea.csv (una fila por compósito del período de
# producción) y la tabla por pantalla.
#
#   PI_GATEWAY_HOST=<ip> PI_GATEWAY_PORT=5173 pixi run python scripts/en_linea.py            # un ciclo
#   PI_GATEWAY_HOST=<ip> PI_GATEWAY_PORT=5173 pixi run python scripts/en_linea.py --cada 20  # en bucle
#   pixi run python scripts/en_linea.py --sin-extraer                                        # sin PI (desarrollo)
#
# La IP de la pasarela cambia: va solo por entorno, nunca en un archivo.
# Sin token: from_pi.cargar_token() devuelve "" y la pasarela lo acepta.
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import subprocess
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.pipeline.config import (ESTIMADO, LIMPIO_COMPOSITO, DATA_FINAL, RAW_COURIER_PI,
                                 RAW_COMPOSITO, VENTANA_COMPOSITO_H, FECHA_INICIO_PRODUCCION,
                                 MIN_BLOQUES_VENTANA, BLOQUE_RESAMPLE)
from src.evaluacion.evaluar import (LEYES_COMPOSITO, mascara_lecturas, alinear_composito,
                                    kalman_serie, elegir_q, periodo_de)

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ETAPAS = ["01a_limpieza_fuentes", "01b_unificacion", "01c_limpieza", "02_features",
          "03_escalado", "04_clusterizado", "06_inferencia_hibrida"]   # sin la 5: bundle congelado
SALIDA = DATA_FINAL / "en_linea.csv"
SOLAPE = pd.Timedelta(hours=2)                       # re-extrae las últimas 2 h: el PI re-archiva tarde
MEDIA_H = pd.Timedelta(hours=VENTANA_COMPOSITO_H / 2)


class MonitorEnLinea:
    """Un ciclo de estimación en línea. `extraer=False` es el modo desarrollo:
    usa los crudos que ya hay en disco y no toca la pasarela."""

    def __init__(self, extraer: bool = True):
        self.extraer = extraer

    # --- 1. adquisición incremental -------------------------------------
    def _unir(self, ruta, nuevo: pd.DataFrame) -> tuple[int, str]:
        viejo = pd.read_csv(ruta)
        nuevo = nuevo.assign(ts=nuevo["ts"].astype(str))
        u = (pd.concat([viejo, nuevo]).drop_duplicates()
             .sort_values("ts", kind="stable").reset_index(drop=True))
        u.to_csv(ruta, index=False)
        return len(u) - len(viejo), u["ts"].max()

    def adquirir(self) -> None:
        from src.acquisition.from_pi import extraer_courier, extraer_composito
        for ruta, fn in [(RAW_COURIER_PI, extraer_courier), (RAW_COMPOSITO, extraer_composito)]:
            desde = (pd.Timestamp(pd.read_csv(ruta, usecols=["ts"])["ts"].max()) - SOLAPE)
            n, hasta = self._unir(ruta, fn(desde.strftime("%Y-%m-%d %H:%M:%S"), None))
            print(f"  {ruta.name}: +{n} filas, hasta {hasta}")

    # --- 2. pipeline con el bundle congelado ----------------------------
    def procesar(self) -> None:
        for etapa in ETAPAS:
            r = subprocess.run([sys.executable, f"notebooks/{etapa}.py"], cwd=RAIZ,
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"etapa {etapa} falló:\n{r.stderr[-2000:]}")

    # --- 3-4. ventanas pendientes, Kalman y comparación -----------------
    def estimar(self) -> pd.DataFrame:
        est = pd.read_csv(ESTIMADO, parse_dates=["ts"])
        comp = pd.read_csv(LIMPIO_COMPOSITO, parse_dates=["ts"])
        ultima = est.loc[est["fuente"] != "bd", "ts"].max()

        mascara = mascara_lecturas(est)
        A = alinear_composito(est, comp, mascara)        # publicados: igual que la etapa 8

        # Compósitos que el laboratorio aún no publica, hasta el turno en curso.
        # Se arman aquí y no con alinear_composito, que descarta los que tienen
        # el centro después del último dato (correcto en la 8, no en línea). Una
        # ventana sin datos queda con la estimación en NaN y n_bloques = 0.
        cols = [f"{l}_est" for l in LEYES_COMPOSITO]
        bloques = est.loc[mascara].set_index("ts")[cols].resample(BLOQUE_RESAMPLE).mean().dropna()
        marca, ahora, pend = comp["ts"].max() + pd.Timedelta(hours=12), pd.Timestamp.now(), []
        while marca - MEDIA_H <= ahora:
            v = bloques.loc[marca - MEDIA_H: marca + MEDIA_H]
            fila = {"ts": marca, "turno": "dia" if marca.hour < 12 else "noche", "n_bloques": len(v)}
            fila.update(v.mean().to_dict() if len(v) >= MIN_BLOQUES_VENTANA else {c: np.nan for c in cols})
            pend.append(fila)
            marca += pd.Timedelta(hours=12)
        if pend:
            A = pd.concat([A, pd.DataFrame(pend)], ignore_index=True)
            A["periodo"] = periodo_de(A["ts"])
        for ley in LEYES_COMPOSITO:                      # q elegido SOLO con train, como en la etapa 8
            q, _ = elegir_q(A, ley)
            A[f"{ley}_kal"], _ = kalman_serie(A, ley, q)
        A["ventana_completa"] = A["ts"] + MEDIA_H <= pd.Timestamp.now()
        A["horas_sin_lectura"] = (pd.Timestamp.now() - ultima) / pd.Timedelta(hours=1)
        A["publicado"] = A["pCu"].notna()
        A = A[A["ts"] >= pd.Timestamp(FECHA_INICIO_PRODUCCION)].reset_index(drop=True)
        A["ultima_lectura"] = ultima
        A.to_csv(SALIDA, index=False)
        return A

    def ciclo(self) -> pd.DataFrame:
        t0 = time.time()
        print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] ciclo en línea")
        if self.extraer:
            self.adquirir()
        self.procesar()
        A = self.estimar()
        h = A["horas_sin_lectura"].iloc[0]
        print(f"  listo en {time.time() - t0:.0f} s; última lectura del courier: {A['ultima_lectura'].iloc[0]}"
              + (f"  ** SIN LECTURAS NUEVAS HACE {h:.1f} h **" if h > 1 else ""))
        mostrar(A)
        return A


def mostrar(A: pd.DataFrame, n: int = 6) -> None:
    """Últimos n compósitos: estimación en escala balde (Kalman) contra el lab."""
    filas = []
    for f in A.tail(n).itertuples(index=False):
        fila = {"compósito": f.ts.strftime("%m-%d %H:%M"), "bloques": f"{f.n_bloques}/{2 * VENTANA_COMPOSITO_H + 1}",
                "estado": "lab" if f.publicado else ("ventana cerrada" if f.ventana_completa else "en curso")}
        for ley, el in [("pCu", "Cu"), ("pFe", "Fe"), ("pMo", "Mo")]:
            est, lab = getattr(f, f"{ley}_kal"), getattr(f, ley)
            if np.isnan(est):
                fila[el] = "sin datos"
            else:
                fila[el] = f"{est:6.2f}" if np.isnan(lab) else f"{est:6.2f} vs {lab:6.2f} ({est - lab:+.2f})"
        filas.append(fila)
    print(pd.DataFrame(filas).to_string(index=False))


def _main():
    ap = argparse.ArgumentParser(description="Estimación en línea de Cu/Fe/Mo vs compósito")
    ap.add_argument("--cada", type=float, default=0, help="minutos entre ciclos (0 = un solo ciclo)")
    ap.add_argument("--sin-extraer", action="store_true", help="no consulta el PI (modo desarrollo)")
    args = ap.parse_args()
    monitor = MonitorEnLinea(extraer=not args.sin_extraer)
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
