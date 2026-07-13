# ============================================================
# src/evaluacion/evaluar.py — Evaluación vs laboratorio químico REAL
# ============================================================
# Unifica las celdas 08 y 09 en una sola función parametrizable. Corre la
# inferencia sobre las intensidades de un período, promedia en ventanas de 12 h
# CENTRADAS en el timestamp del ensayo, excluye ventanas con fuga de datos y
# calcula R2, MAE y correlación contra el laboratorio real.
# ============================================================

import numpy as np                                        # cálculo numérico
import pandas as pd                                        # DataFrames
from sklearn.metrics import r2_score, mean_absolute_error  # métricas

from src.pipeline.config import CANALES                    # canales de intensidad
from src.pipeline.inferencia import EstimadorHibrido       # estimador (import normal, sin exec)

MESES = {"ene": "01", "feb": "02", "mar": "03", "abr": "04", "may": "05", "jun": "06",
         "jul": "07", "ago": "08", "sep": "09", "oct": "10", "nov": "11", "dic": "12"}


def parse_fecha_es(s: str) -> pd.Timestamp:
    """Convierte '01-ene-24 06:00:00' (fecha en español) a Timestamp."""
    d, t = s.split(" ")                                   # separa fecha y hora
    dd, mm, yy = d.split("-")                             # día, mes (texto), año
    return pd.Timestamp(f"20{yy}-{MESES[mm]}-{dd} {t}")   # arma el timestamp


def cargar_lab_real(ruta: str) -> pd.DataFrame:
    """Carga el laboratorio real (assay_lab_raw.csv) y deja 1 valor por 12 h."""
    lab = pd.read_csv(ruta)                               # lee el CSV
    lab["ts"] = lab["date_time"].apply(parse_fecha_es)    # parsea la fecha en español
    lab = lab.drop_duplicates(subset=["Cu", "Mo", "Fe"], keep="first")  # 1 assay por 12 h
    return lab.sort_values("ts")                          # ordenado por tiempo


def evaluar_periodo(inten: pd.DataFrame, lab: pd.DataFrame, cal: pd.DataFrame,
                    fecha_ini: str, fecha_fin: str, estimador=None) -> pd.DataFrame:
    """Evalúa el pipeline en un período contra el laboratorio real.

    Parámetros
    ----------
    inten : intensidades del stream (con columnas date, time y los canales).
    lab : laboratorio real ya cargado con columna 'ts' (ver cargar_lab_real).
    cal : dataset de calibración (para detectar fuga).
    fecha_ini, fecha_fin : límites del período (strings 'YYYY-MM-DD').
    estimador : EstimadorHibrido ya instanciado; si None, se crea uno.

    Retorna
    -------
    M : DataFrame con las ventanas comparadas (estimado vs real por elemento).
    """
    est = estimador or EstimadorHibrido()                 # reutiliza o crea el estimador

    # --- Filtrar intensidades al período y marcar fuga (muestras de calibración) ---
    inten = inten.copy()                                  # no mutar el original
    inten["ts"] = pd.to_datetime(inten["date"].astype(str) + " " + inten["time"])  # timestamp
    inten = inten[(inten["ts"] >= fecha_ini) & (inten["ts"] < fecha_fin)]          # recorta período
    cal_keys = set(map(tuple, cal[CANALES].round(3).values))                       # firmas de calibración
    inten = inten.assign(es_calib=[tuple(np.round(r, 3)) in cal_keys               # marca fuga
                                   for r in inten[CANALES].values])

    # --- Inferencia sobre cada lectura ---
    filas = []                                            # estimaciones por lectura
    for _, f in inten.iterrows():                         # recorre las lecturas
        res = est.predecir({c: f[c] for c in CANALES})    # estima
        if res["leyes"]:                                  # si pasó la compuerta de calidad
            filas.append({"ts": f["ts"], "es_calib": f["es_calib"], **res["leyes"]})  # guarda
    pred = pd.DataFrame(filas).set_index("ts").sort_index()  # serie temporal de estimaciones
    pred = pred[~pred["es_calib"]]                        # excluye lecturas de calibración (fuga)

    # --- Alinear en ventanas de 12 h CENTRADAS en el timestamp del lab ---
    lab_p = lab[(lab["ts"] >= fecha_ini) & (lab["ts"] < fecha_fin)]  # lab del período
    rows = []                                             # comparaciones
    for _, L in lab_p.iterrows():                         # por cada ensayo de lab
        c = L["ts"]                                       # centro de la ventana (lag 0)
        w = pred[(pred.index > c - pd.Timedelta("6h")) &  # ventana +-6h
                 (pred.index <= c + pd.Timedelta("6h"))]
        if len(w) == 0:                                   # sin intensidades en la ventana
            continue
        rows.append({"fecha": c,                          # promedio estimado vs ley real
                     "pCu": w["pCu"].mean(), "Cu": L["Cu"],
                     "pFe": w["pFe"].mean(), "Fe": L["Fe"],
                     "pMo": w["pMo"].mean(), "Mo": L["Mo"]})
    return pd.DataFrame(rows)                             # tabla de comparación


def reportar_metricas(M: pd.DataFrame) -> pd.DataFrame:
    """Calcula R2, MAE y correlación por elemento y las imprime."""
    filas = []                                            # una fila por elemento
    for est_c, lab_c, nombre in [("pCu", "Cu", "Cu"), ("pFe", "Fe", "Fe"), ("pMo", "Mo", "Mo")]:
        filas.append({"elemento": nombre,
                      "R2": r2_score(M[lab_c], M[est_c]),           # R2
                      "MAE": mean_absolute_error(M[lab_c], M[est_c]),  # error absoluto medio
                      "corr": M[est_c].corr(M[lab_c])})             # correlación
    tabla = pd.DataFrame(filas)                           # tabla de métricas
    print(tabla.round(3).to_string(index=False))         # imprime
    return tabla                                          # devuelve la tabla
