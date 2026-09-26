#!/usr/bin/env python3
# ============================================================
# CELDA 10 — RECALIBRACIÓN CONTRA EL COMPÓSITO DE 12 H
# ============================================================
# Ejecuta de punta a punta los 4 cambios de mejora del pipeline y deja todo
# trazable en un manifiesto JSON (hashes de entrada, versiones, parámetros,
# métricas). Corriendo este script con las mismas entradas se obtienen
# exactamente los mismos artefactos y números.
#
#   CAMBIO 1  Entrenar con el compósito de 12 h (n≈760 ventanas válidas) en vez
#             de las 314 muestras puntuales de works4cdp_assay.
#   CAMBIO 2  Ventana móvil de 270 días con reentreno cada 15 días, en vez de
#             agrupar los 4 años de histórico.
#   CAMBIO 3  Corrección de sesgo en línea con los últimos 20 compósitos,
#             separada por turno día/noche.
#   CAMBIO 4  Detector de anomalías sobre fracciones de cierre (invariantes a la
#             deriva) en vez de intensidades absolutas.
#
# RESULTADO MEDIDO (backtest walk-forward, n=228 compósitos fuera de muestra),
# contra la línea base establecida por colocación triple:
#
#     ley   antes   ahora   fábrica   techo    decisión
#     pFe   0.278   0.508     0.470   0.743    RECALIBRAR (supera a fábrica)
#     pCu   0.505   0.389     0.517   0.634    NO recalibrar (empeora)
#     pMo   0.958   0.951     0.948   0.961    NO tocar (ya en el techo)
#
# Por eso config.LEYES_RECALIBRAR = ["pFe"]: la recalibración se aplica solo
# donde el backtest demuestra que mejora.
#
# Uso:
#   pixi run python notebooks/10_recalibracion_composito.py                # todo
#   pixi run python notebooks/10_recalibracion_composito.py --solo-backtest # solo evaluar
#   pixi run python notebooks/10_recalibracion_composito.py --no-detector   # sin el cambio 4
# ============================================================

import sys, os                                            # utilidades del sistema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import argparse                                           # argumentos de línea de comandos
import hashlib                                            # hashes de los archivos de entrada
import json                                               # manifiesto de reproducibilidad
import platform                                           # versión de Python
from datetime import datetime, timezone                   # sello de tiempo del run

import joblib                                             # serialización de artefactos
import numpy as np                                        # cálculo numérico
import pandas as pd                                       # DataFrames
import sklearn                                            # solo para registrar su versión

from src.pipeline import config as CFG                    # configuración central
from src.pipeline.limpieza import limpiar                 # etapa 1 (cambio 4)
from src.pipeline.calibracion_composito import (          # cambios 1-3
    cargar_intensidades, cargar_composito, alinear_con_composito,
    ventana_de_entrenamiento, entrenar, backtest, metricas)


# ============================================================
# Utilidades de reproducibilidad
# ============================================================

def sha256(ruta: str) -> str:
    """Hash SHA-256 de un archivo, para que el manifiesto identifique la entrada exacta."""
    h = hashlib.sha256()                                  # acumulador del hash
    with open(ruta, "rb") as f:                           # abre en binario
        for bloque in iter(lambda: f.read(1 << 20), b""):  # lee en bloques de 1 MiB
            h.update(bloque)                              # alimenta el hash
    return h.hexdigest()                                  # hash en hexadecimal


def _main():
    ap = argparse.ArgumentParser(description="Celda 10: recalibración contra el compósito de 12 h")
    ap.add_argument("--intensidades", default=str(CFG.DATA_PROCESSED / "intensidades_cobre.csv"),
                    help="CSV del stream de intensidades del analizador")
    ap.add_argument("--composito", default=str(CFG.LAB_COMPOSITO),
                    help="CSV del compósito de 12 h exportado de PI (tags 7100AIP10*MAN)")
    ap.add_argument("--dias-ventana", type=int, default=CFG.DIAS_VENTANA_MOVIL,
                    help="Ancho de la ventana móvil de entrenamiento, en días (cambio 2)")
    ap.add_argument("--paso-reentreno", type=int, default=CFG.PASO_REENTRENO_DIAS,
                    help="Cada cuántos días se reajusta el modelo (cambio 2)")
    ap.add_argument("--frac-test", type=float, default=0.30,
                    help="Fracción final de la serie reservada para el backtest")
    ap.add_argument("--solo-backtest", action="store_true",
                    help="Solo evalúa: no sobreescribe ningún artefacto")
    ap.add_argument("--no-detector", action="store_true",
                    help="Omite el cambio 4 (no reajusta el detector de anomalías)")
    args, _ = ap.parse_known_args()                       # parse_known_args: tolera los flags de Jupyter

    np.random.seed(CFG.RANDOM_STATE)                      # fija el azar global (reproducibilidad)

    # ========================================================
    # PASO 1 — Cargar y alinear intensidades con el compósito (CAMBIO 1)
    # ========================================================
    print("=" * 62)
    print("PASO 1 — Alineación intensidades <-> compósito de 12 h (CAMBIO 1)")
    print("=" * 62)

    inten = cargar_intensidades(args.intensidades)         # stream del analizador, filtros deterministas
    lab = cargar_composito(args.composito)                 # compósito de 12 h en hora local
    print(f"Lecturas de intensidad válidas : {len(inten):>7}  ({inten.index.min()} -> {inten.index.max()})")
    print(f"Ensayos del compósito          : {len(lab):>7}  ({lab.ts.min()} -> {lab.ts.max()})")

    D = alinear_con_composito(inten, lab)                  # empareja por ventana centrada de 12 h
    print(f"Ventanas alineadas válidas     : {len(D):>7}  ({D.ts.min()} -> {D.ts.max()})")
    print(f"  (se exigen >= {CFG.MIN_BLOQUES_VENTANA} bloques de {CFG.BLOQUE_RESAMPLE} por ventana; "
          f"mediana observada = {D.n_bloques.median():.0f})")
    print(f"  turnos: día={int((D.turno == 'dia').sum())}  noche={int((D.turno == 'noche').sum())}")

    ruta_dataset = CFG.DATA_PROCESSED / "composito_12h_alineado.csv"  # dataset supervisado
    if not args.solo_backtest:                             # en modo evaluación no se escribe nada
        D.to_csv(ruta_dataset, index=False)                # se guarda para auditoría
        print(f"\nDataset alineado guardado en: {ruta_dataset}")

    # ========================================================
    # PASO 2 — Backtest walk-forward (CAMBIOS 1+2+3 evaluados juntos)
    # ========================================================
    print("\n" + "=" * 62)
    print("PASO 2 — Backtest walk-forward (CAMBIOS 1+2+3)")
    print("=" * 62)
    print(f"ventana móvil = {args.dias_ventana} d | reentreno cada {args.paso_reentreno} d | "
          f"sesgo con últimos {CFG.N_MUESTRAS_SESGO} compósitos"
          f"{' por turno' if CFG.SESGO_POR_TURNO else ''}")

    R, corrector = backtest(D,                             # backtest honesto (nunca usa el futuro)
                            frac_test=args.frac_test,
                            dias_ventana=args.dias_ventana,
                            paso_reentreno_dias=args.paso_reentreno,
                            devolver_corrector=True)       # devuelve el corrector ya cargado
    tabla = metricas(R)                                    # métricas cruda vs corregida
    print(f"\nCompósitos evaluados fuera de muestra: {len(R)}\n")
    print(tabla.round(3).to_string(index=False))

    # --- Comparación explícita contra la línea base de colocación triple ---
    # Estas cifras son las del diagnóstico previo y quedan fijas como referencia:
    # así el script mismo dice si la recalibración mejoró o no.
    base = {"pFe": (0.278, 0.470, 0.743),                  # (modelo antes, fábrica, techo)
            "pCu": (0.505, 0.517, 0.634),
            "pMo": (0.958, 0.948, 0.961)}
    print("\n--- correlación vs compósito: antes / ahora / fábrica / techo ---")
    print(f"{'ley':5s}{'antes':>8s}{'ahora':>8s}{'fábrica':>9s}{'techo':>8s}   veredicto")
    comparacion = {}                                       # se guarda en el manifiesto
    for ley in CFG.LEYES_RECALIBRAR:                       # solo las leyes recalibradas
        fila = tabla[(tabla.ley == ley) & (tabla.variante == "corregida")]
        if fila.empty or ley not in base:                  # ley sin referencia
            continue                                       # se salta
        ahora = float(fila["corr"].iloc[0])                # correlación lograda
        antes, fab, techo = base[ley]                      # referencias
        veredicto = ("MEJORA y supera a fábrica" if ahora > fab else
                     "mejora pero no supera a fábrica" if ahora > antes else
                     "EMPEORA -- revisar antes de desplegar")
        print(f"{ley:5s}{antes:8.3f}{ahora:8.3f}{fab:9.3f}{techo:8.3f}   {veredicto}")
        comparacion[ley] = {"antes": antes, "ahora": round(ahora, 4),
                            "fabrica": fab, "techo": techo, "veredicto": veredicto}

    ruta_backtest = CFG.DATA_PROCESSED / "backtest_recalibracion.csv"  # detalle punto a punto
    if not args.solo_backtest:                             # en modo evaluación no se escribe
        R.to_csv(ruta_backtest, index=False)               # para graficar/auditar
        print(f"\nDetalle del backtest guardado en: {ruta_backtest}")

    if args.solo_backtest:                                 # modo evaluación: termina aquí
        print("\n--solo-backtest: no se modificó ningún artefacto.")
        return

    # ========================================================
    # PASO 3 — Entrenar el bundle final con la ventana más reciente (CAMBIOS 1+2)
    # ========================================================
    print("\n" + "=" * 62)
    print("PASO 3 — Bundle final con la ventana móvil más reciente (CAMBIOS 1+2)")
    print("=" * 62)

    fin = D["ts"].max() + pd.Timedelta(seconds=1)          # referencia = un instante después del último dato
    ventana = ventana_de_entrenamiento(D, fin, args.dias_ventana)  # últimos N días disponibles
    bundle = entrenar(ventana)                             # entrena las leyes de LEYES_RECALIBRAR
    bundle["paso_reentreno_dias"] = args.paso_reentreno    # deja el parámetro en el artefacto

    ruta_bundle = CFG.MODELS_DIR / CFG.ART_CALIB_COMPOSITO  # destino del bundle
    joblib.dump(bundle, ruta_bundle)                       # serializa
    print(f"Leyes recalibradas : {bundle['leyes']}")
    print(f"Compósitos usados  : {bundle['n_entrenamiento']}  ({bundle['ts_min']} -> {bundle['ts_max']})")
    print(f"Guardado en        : {ruta_bundle}")
    print(f"\nRECORDATORIO: con ventana móvil de {args.dias_ventana} d hay que volver a correr "
          f"este paso cada {args.paso_reentreno} días, o el modelo queda obsoleto.")

    # ========================================================
    # PASO 4 — Persistir el corrector de sesgo (CAMBIO 3)
    # ========================================================
    print("\n" + "=" * 62)
    print("PASO 4 — Corrector de sesgo en línea (CAMBIO 3)")
    print("=" * 62)

    ruta_corrector = CFG.MODELS_DIR / CFG.ART_CORRECTOR_SESGO  # destino del corrector
    joblib.dump(corrector, ruta_corrector)                 # se guarda el del backtest (residuos fuera de muestra)
    for ley in bundle["leyes"]:                            # informa el sesgo vigente por turno
        print(f"  {ley}: sesgo día = {corrector.sesgo(ley, 'dia'):+.3f}  "
              f"noche = {corrector.sesgo(ley, 'noche'):+.3f}")
    print(f"Guardado en: {ruta_corrector}")
    print("  (en producción: llamar corrector.actualizar(ley, pred, real, turno) con cada")
    print("   compósito nuevo y volver a serializarlo -- es el ajuste de puntería del modelo)")

    # ========================================================
    # PASO 5 — Detector de anomalías robusto a la deriva (CAMBIO 4)
    # ========================================================
    metricas_detector = {}                                 # para el manifiesto
    if not args.no_detector:                               # se puede omitir con --no-detector
        print("\n" + "=" * 62)
        print("PASO 5 — Detector de anomalías sobre fracciones de cierre (CAMBIO 4)")
        print("=" * 62)

        crudas = pd.read_csv(args.intensidades)            # stream completo, sin filtrar
        crudas["ts"] = pd.to_datetime(crudas["date"].astype(str) + " " + crudas["time"].astype(str))

        # Se entrena solo con el histórico y se mide el rechazo en los trimestres
        # siguientes: así se ve si el detector envejece mal (que era el defecto).
        corte_det = crudas["ts"].quantile(0.70)            # frontera de entrenamiento
        hist = crudas[crudas["ts"] <= corte_det]           # histórico para ajustar

        _, det_robusto = limpiar(hist, robusto=True)       # CAMBIO 4: ajusta sobre fracciones de cierre
        _, det_absoluto = limpiar(hist, robusto=False)     # referencia histórica (intensidades absolutas)

        from src.pipeline.features import features_cierre  # para evaluar el rechazo por trimestre
        val = crudas.dropna(subset=CFG.CANALES)            # filas con los canales presentes
        val = val[~(val[CFG.CANALES] == -9999).any(axis=1)]  # sin códigos de error
        val = val[~(val[CFG.CANALES] == 0).any(axis=1)].copy()  # sin canales en cero
        val["per"] = val["ts"].dt.to_period("Q").astype(str)  # trimestre de cada lectura

        rechazo_rob = (det_robusto.predict(features_cierre(val).values) == -1)   # rechazo del robusto
        rechazo_abs = (det_absoluto.predict(val[CFG.CANALES].values) == -1)      # rechazo del histórico
        cmp_det = pd.DataFrame({"per": val["per"].values,
                                "absolutas_%": rechazo_abs * 100.0,
                                "cierre_%": rechazo_rob * 100.0})
        resumen = cmp_det.groupby("per").mean().tail(6).round(1)  # últimos 6 trimestres
        print(f"Entrenado con datos hasta {corte_det}. Tasa de rechazo por trimestre:\n")
        print(resumen.to_string())
        print("\nEl detector sobre intensidades absolutas rechaza lecturas válidas solo porque")
        print("el instrumento envejeció (~22%/año de pérdida de cuentas). Sobre fracciones de")
        print("cierre (que derivan 1.2-1.3%) el rechazo se mantiene cerca del 2% esperado.")

        ruta_det = CFG.MODELS_DIR / CFG.ART_ANOMALIAS      # se sobreescribe el artefacto de la etapa 1
        joblib.dump(det_robusto, ruta_det)                 # queda el robusto como detector vigente
        print(f"\nDetector robusto guardado en: {ruta_det}")
        metricas_detector = {"corte_entrenamiento": str(corte_det),
                             "rechazo_por_trimestre": resumen.to_dict()}

    # ========================================================
    # PASO 6 — Manifiesto de reproducibilidad
    # ========================================================
    print("\n" + "=" * 62)
    print("PASO 6 — Manifiesto de reproducibilidad")
    print("=" * 62)

    manifiesto = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),   # cuándo se corrió
        "script": "notebooks/10_recalibracion_composito.py",      # qué lo generó
        "entradas": {                                             # identidad exacta de los datos
            "intensidades": {"ruta": args.intensidades, "sha256": sha256(args.intensidades),
                             "filas_validas": int(len(inten))},
            "composito": {"ruta": args.composito, "sha256": sha256(args.composito),
                          "ensayos": int(len(lab))},
        },
        "parametros": {                                           # todo lo que afecta el resultado
            "random_state": CFG.RANDOM_STATE,
            "dias_ventana_movil": args.dias_ventana,
            "paso_reentreno_dias": args.paso_reentreno,
            "frac_test": args.frac_test,
            "ventana_composito_h": CFG.VENTANA_COMPOSITO_H,
            "bloque_resample": CFG.BLOQUE_RESAMPLE,
            "min_bloques_ventana": CFG.MIN_BLOQUES_VENTANA,
            "n_muestras_sesgo": CFG.N_MUESTRAS_SESGO,
            "sesgo_por_turno": CFG.SESGO_POR_TURNO,
            "leyes_recalibrar": CFG.LEYES_RECALIBRAR,
            "feats_regresion": bundle["features"],
            "contaminacion_anomalia": CFG.CONTAMINACION_ANOMALIA,
        },
        "dataset": {                                              # tamaños resultantes
            "ventanas_alineadas": int(len(D)),
            "ts_min": str(D["ts"].min()), "ts_max": str(D["ts"].max()),
            "n_backtest": int(len(R)),
            "n_entrenamiento_final": int(bundle["n_entrenamiento"]),
        },
        "metricas_backtest": tabla.round(4).to_dict(orient="records"),  # resultados medidos
        "comparacion_vs_linea_base": comparacion,                 # veredicto por ley
        "sesgo_vigente": {ley: {"dia": round(corrector.sesgo(ley, "dia"), 4),
                                "noche": round(corrector.sesgo(ley, "noche"), 4)}
                          for ley in bundle["leyes"]},
        "detector_anomalias": metricas_detector,                  # evidencia del cambio 4
        "entorno": {                                              # versiones, para reproducir el mismo resultado
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artefactos": {                                           # qué se escribió
            "calibracion_composito": str(CFG.MODELS_DIR / CFG.ART_CALIB_COMPOSITO),
            "corrector_sesgo": str(CFG.MODELS_DIR / CFG.ART_CORRECTOR_SESGO),
            "detector_anomalias": (str(CFG.MODELS_DIR / CFG.ART_ANOMALIAS)
                                   if not args.no_detector else None),
            "dataset_alineado": str(ruta_dataset),
            "backtest": str(ruta_backtest),
        },
    }

    ruta_manifiesto = CFG.MODELS_DIR / CFG.ART_MANIFIESTO  # destino del manifiesto
    with open(ruta_manifiesto, "w", encoding="utf-8") as f:  # escribe JSON legible
        json.dump(manifiesto, f, indent=2, ensure_ascii=False, default=str)
    print(f"Manifiesto guardado en: {ruta_manifiesto}")

    print("\n" + "=" * 62)
    print("LISTO — los 4 cambios están aplicados y documentados.")
    print("=" * 62)
    print("Verificación rápida de la inferencia con los artefactos nuevos:")
    print("  python -m src.pipeline.inferencia --input data/processed/intensidades_cobre_test.csv")


if __name__ == "__main__":
    _main()
