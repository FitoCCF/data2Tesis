#!/usr/bin/env python3
# ============================================================
# src/pipeline/limpieza.py — Etapa 1c: limpieza general sobre la tabla maestra
# ============================================================
# Entrada: courier_unificado.csv (etapa 1b), una fila por lectura real del
# courier. Las reglas deterministas por fuente ya se aplicaron en la 1a; aquí
# van las que miran la tabla completa:
#
#   congelada  vector de los 4 metales idéntico a la lectura anterior. Tras
#              la 1a no hay ninguna (medido 2026-09-30); se mantiene como
#              resguardo para datos nuevos y de producción.
#   anomalia   IsolationForest sobre fracciones de cierre m/Σm (robustas a la
#              deriva del instrumento, ~22%/año de cuentas: sobre intensidades
#              absolutas rechazaba 16% del trimestre corriente). Se AJUSTA solo
#              con lecturas hasta FECHA_CORTE_TRAIN, para que el detector no
#              conozca el período de prueba; se APLICA a todas.
#   valida     ~anomalia & ~congelada & ~sostenida: lecturas para AJUSTAR las
#              etapas no supervisadas (ortho, escalado, GMM).
#   valida_ley con ley & ~congelada & ~sostenida & sin ley sospechosa: muestras
#              para los modelos. La anomalía NO excluye aquí (ver limpiar()).
#              n6sc_desfasado y hora_bd_desfasada no afectan a ninguna de las
#              dos: la lectura se reconstruyó bien.
#   periodo    'train' (ts <= FECHA_CORTE_TRAIN), 'test' (hasta antes de
#              FECHA_INICIO_PRODUCCION) o 'produccion'. Corte único del
#              pipeline: toda etapa que ajusta algo usa solo 'train'.
#
# No borra filas: marca. Las etapas siguientes filtran por 'valida'.
#
# Ejecutable de forma independiente:
#   python -m src.pipeline.limpieza                                   # entrena y marca
#   python -m src.pipeline.limpieza --input nuevo.csv --artifact-in models/anomaly_detector.joblib
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import RANDOM_STATE, CONTAMINACION_ANOMALIA, FECHA_CORTE_TRAIN, FECHA_INICIO_PRODUCCION
from .features import features_cierre

METALES = ["n1fe", "n2cu", "n3zn", "n4mo"]
LEYES = ["pFe", "pCu", "pZn", "pMo", "pIns", "pSol"]


def fin_de_train(corte: str = FECHA_CORTE_TRAIN) -> pd.Timestamp:
    """Último instante del período de entrenamiento: el día de corte entero."""
    return pd.Timestamp(corte) + pd.Timedelta(days=1)


def limpiar(df: pd.DataFrame, detector: IsolationForest | None = None,
            corte: str = FECHA_CORTE_TRAIN) -> tuple[pd.DataFrame, IsolationForest, dict]:
    """Marca congeladas, anomalías y validez sobre la tabla maestra.

    detector=None -> modo ENTRENAMIENTO: ajusta un IsolationForest con las
    lecturas completas del período 'train'. Con un detector -> modo
    INFERENCIA: lo aplica tal cual.

    Retorna (df con columnas periodo/congelada/anomalia/valida/valida_ley, detector, resumen).
    """
    df = df.sort_values("ts").reset_index(drop=True).copy()
    df["periodo"] = np.select([df["ts"] < fin_de_train(corte),
                               df["ts"] < pd.Timestamp(FECHA_INICIO_PRODUCCION)],
                              ["train", "test"], default="produccion")

    df["congelada"] = (df[METALES] == df[METALES].shift()).all(axis=1)

    x = features_cierre(df)
    evaluable = x.notna().all(axis=1)
    if detector is None:
        detector = IsolationForest(n_estimators=200, contamination=CONTAMINACION_ANOMALIA,
                                   random_state=RANDOM_STATE)
        detector.fit(x[evaluable & (df["periodo"] == "train")])
        detector.corte_train_ = corte                      # queda registrado en el artefacto
    df["anomalia"] = False
    df.loc[evaluable, "anomalia"] = detector.predict(x[evaluable]) == -1

    sostenida = df["sostenida"].fillna(False).astype(bool) if "sostenida" in df else False
    ley_sospechosa = df["leyes_sospechosas"].notna() if "leyes_sospechosas" in df else False
    con_ley = df[[c for c in LEYES if c in df]].notna().any(axis=1)
    # Dos usos, dos criterios:
    #  valida       lecturas para AJUSTAR lo no supervisado (ortho, escalado, GMM):
    #               sobran lecturas, así que la anomalía sí excluye.
    #  valida_ley   muestras con ley para los modelos: la anomalía NO excluye
    #               (el 2% de contamination es arbitrario y las muestras con ley
    #               son lo escaso); su control de calidad es la ley sospechosa,
    #               que ya contrasta la ley con su canal del courier.
    df["valida"] = ~df["anomalia"] & ~df["congelada"] & ~sostenida
    df["valida_ley"] = con_ley & ~df["congelada"] & ~sostenida & ~ley_sospechosa

    tasa = df.groupby("periodo")["anomalia"].mean().mul(100).round(2).to_dict()
    r = {"filas": len(df),
         "por_periodo": df["periodo"].value_counts().to_dict(),
         "congeladas": int(df["congelada"].sum()),
         "no_evaluables_anomalia": int((~evaluable).sum()),
         "anomalias": int(df["anomalia"].sum()),
         "tasa_anomalia_pct": tasa,
         "sostenidas": int(pd.Series(sostenida).sum()),
         "ley_sospechosa": int(pd.Series(ley_sospechosa).sum()),
         "validas": int(df["valida"].sum()),
         "con_ley": int(con_ley.sum()),
         "validas_ley": int(df["valida_ley"].sum()),
         "validas_ley_anomalas": int((df["valida_ley"] & df["anomalia"]).sum())}
    return df, detector, r


def _main():
    import argparse
    import joblib

    from .config import UNIFICADO, LIMPIO, MODELS_DIR, ART_ANOMALIAS

    ap = argparse.ArgumentParser(description="Etapa 1c: limpieza general sobre la tabla maestra")
    ap.add_argument("--input", default=str(UNIFICADO))
    ap.add_argument("--output", default=str(LIMPIO))
    ap.add_argument("--corte", default=FECHA_CORTE_TRAIN, help="Último día del período de entrenamiento")
    ap.add_argument("--artifact-in", default=None, help="Detector ya entrenado (modo INFERENCIA)")
    ap.add_argument("--artifact-out", default=str(MODELS_DIR / ART_ANOMALIAS))
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["ts"])
    detector = joblib.load(args.artifact_in) if args.artifact_in else None
    df, detector, r = limpiar(df, detector, corte=args.corte)
    df.to_csv(args.output, index=False)
    print(f"{args.input} -> {args.output}")
    for k, v in r.items():
        print(f"  {k}: {v}")
    if args.artifact_in is None:
        joblib.dump(detector, args.artifact_out)
        print(f"Detector guardado en: {args.artifact_out}")


if __name__ == "__main__":
    _main()
