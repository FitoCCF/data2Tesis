# ============================================================
# src/evaluacion/descripcion.py — qué modelo corre cada monitor en línea
# ============================================================
# Se lee del GMM guardado (no de la config) para que el texto diga lo que de
# verdad corre: data2TesisV2 agrupa por Fe, Zn y Mo; AlimConc solo por Mo.
# ============================================================

import joblib

from src.pipeline.config import MODELS_DIR, ART_GMM, ART_REGRESION, PROJECT_ROOT
from src.pipeline.escalado import FEATS_ESCALADAS
from src.pipeline import config

ELEMENTO = {"lr_fe_cu_z": "Fe", "lr_zn_cu_z": "Zn", "lr_mo_cu_z": "Mo"}


def _carpeta() -> str:
    return PROJECT_ROOT.name


def modelo_concentrado() -> str:
    try:
        gmm = joblib.load(MODELS_DIR / ART_GMM)
        k, n = gmm.n_components, gmm.n_features_in_
    except Exception as e:
        return f"Modelo: no se pudo leer {ART_GMM} ({type(e).__name__}) · carpeta {_carpeta()}"
    feats = getattr(config, "FEATS_GMM", FEATS_ESCALADAS)
    if len(feats) != n:                     # el .joblib no es el de esta config
        return f"Modelo: GMM k={k} sobre {n} features (no calza con la config) + Kalman · carpeta {_carpeta()}"
    if n == 1:
        el = ELEMENTO[feats[0]]
        grupos = f"solo {el} (lr_{el.lower()}_cu, grupos {el}1…{el}{k})"
    else:
        els = [ELEMENTO[f] for f in feats]
        grupos = f"{', '.join(els[:-1])} y {els[-1]} (log-cocientes contra Cu)"
    try:                                    # qué celdas (ley, grupo) pasaron el control de la etapa 5
        ruteo = joblib.load(MODELS_DIR / ART_REGRESION)["tabla_ruteo"]
        locales = [f"{ley[1:]}-{g}" for (ley, g), v in ruteo.items() if v == "local"]
    except Exception:
        locales = None
    if locales == []:
        uso = "ningún grupo pasó el control: modelo GLOBAL, sin grupos"
    elif locales:
        uso = f"modelo local solo en {', '.join(locales)}; el resto global"
    else:
        uso = "modelo local por grupo"
    return f"Modelo: GMM k={k} sobre {grupos} → {uso} → Kalman · carpeta {_carpeta()}"


def modelo_cabeza(linea: str) -> str:
    return f"Modelo: Ridge global con el courier de cabeza {linea}, sin grupos → Kalman · carpeta {_carpeta()}"
