#!/usr/bin/env python3
# ============================================================
# CELDA 8 — MODELO vs COMPÓSITO DE LABORATORIO (12 h), SIN Y CON KALMAN
# ============================================================
# Compara la estimación del modelo (etapa 6) contra el compósito químico de
# 12 h: media de la estimación en la ventana centrada del corte (bloques de
# 30 min, >= 8 bloques). Después aplica el corrector de sesgo por Kalman, con
# q_por_dia elegido SOLO en los compósitos de train, y compara las métricas.
#
# Métricas solo fuera de muestra (test jun-ago + producción sep); el periodo
# de train se grafica sombreado como contexto. Se reporta sin y con las
# lecturas que el IsolationForest marcó como anómalas.
#
#   courier_estimado.csv + composito_limpio.csv
#       -> data/final/eval_composito.csv, data/final/eval_composito_metricas.csv
#       -> reports/fig_08_modelo_vs_composito.png/.pdf
#
#   pixi run python notebooks/08_evaluacion_laboratorio.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import pandas as pd

from src.pipeline.config import (ESTIMADO, LIMPIO_COMPOSITO, DATA_FINAL, REPORTS_DIR,
                                 FECHA_INICIO_PRODUCCION)
from src.pipeline.limpieza import fin_de_train
from src.evaluacion.evaluar import (LEYES_COMPOSITO, alinear_composito, mascara_lecturas,
                                    kalman_serie, elegir_q, metricas)
from src.evaluacion.figuras import figura_tendencia

est = pd.read_csv(ESTIMADO, parse_dates=['ts'])
comp = pd.read_csv(LIMPIO_COMPOSITO, parse_dates=['ts'])
DATA_FINAL.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

filas, principal = [], None
for con_anom in (False, True):
    A = alinear_composito(est, comp, mascara_lecturas(est, con_anom))
    etiqueta = "con anomalías" if con_anom else "sin anomalías"
    print(f"\n== Compósitos alineados ({etiqueta}): {A.groupby('periodo').size().to_dict()} ==")
    for ley in LEYES_COMPOSITO:
        q, T = elegir_q(A, ley)
        A[f"{ley}_kal"], _ = kalman_serie(A, ley, q)
        if not con_anom:
            print(f"  {ley}: q_por_dia elegido en train = {q}  (R² train {T['R2'].max():.3f})")
        for per, mask in [("test", A.periodo == "test"), ("produccion", A.periodo == "produccion"),
                          ("fuera de muestra", A.periodo != "train")]:
            for var, col in [("modelo", f"{ley}_est"), ("modelo + Kalman", f"{ley}_kal")]:
                filas.append({"lecturas": etiqueta, "ley": ley, "periodo": per, "variante": var,
                              "q_por_dia": q if var != "modelo" else None,
                              **metricas(A.loc[mask, ley], A.loc[mask, col])})
    if not con_anom:
        principal = A

M = pd.DataFrame(filas)
pd.set_option('display.width', 200)
print("\n== Métricas vs compósito (est - lab para el sesgo) ==")
print(M.round(3).to_string(index=False))

principal.to_csv(DATA_FINAL / 'eval_composito.csv', index=False)
M.to_csv(DATA_FINAL / 'eval_composito_metricas.csv', index=False)

# --- Figura: tendencia comparativa, sin anomalías (el criterio principal) ---
fuera = M[(M.lecturas == "sin anomalías") & (M.periodo == "fuera de muestra")]
txt = {ley: {"modelo": fuera[(fuera.ley == ley) & (fuera.variante == "modelo")].iloc[0].to_dict(),
             "kalman": fuera[(fuera.ley == ley) & (fuera.variante == "modelo + Kalman")].iloc[0].to_dict()}
       for ley in LEYES_COMPOSITO}
titulo = "Estimación del modelo vs compósito de laboratorio químico (12 h) — métricas fuera de muestra (jun–sep 2026)"
for nombre, desde in [('fig_08_modelo_vs_composito.png', '2026-03-01'),            # foco: fuera de muestra
                      ('fig_08_modelo_vs_composito_completo.png', None)]:          # todo el PI (desde 2025-07)
    ruta = REPORTS_DIR / nombre
    figura_tendencia(principal, list(LEYES_COMPOSITO), titulo, ruta, txt, desde=desde,
                     inicio_test=fin_de_train(), inicio_prod=pd.Timestamp(FECHA_INICIO_PRODUCCION))
    print(f"Figura: {ruta}")
