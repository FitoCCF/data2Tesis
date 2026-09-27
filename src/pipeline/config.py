# ============================================================
# src/pipeline/config.py — Configuración central del pipeline
# ============================================================
# Centraliza rutas, nombres de columnas y constantes para que ninguna
# etapa tenga valores "mágicos" dispersos. Un solo lugar que editar.
# ============================================================

from pathlib import Path                                  # manejo de rutas multiplataforma

# --- Raíz del proyecto (dos niveles arriba de este archivo: src/pipeline/config.py) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]        # .../data2TesisV2
DATA_RAW = PROJECT_ROOT / "data" / "raw"                  # carpeta de datos crudos
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"      # carpeta de datos procesados
DATA_FINAL = PROJECT_ROOT / "data" / "final"              # resultados finales
MODELS_DIR = PROJECT_ROOT / "models"                      # carpeta de artefactos .joblib
REPORTS_DIR = PROJECT_ROOT / "reports"                    # figuras/tablas de salida

# --- Canales espectrales (n6sc = sólido en la muestra / dilución) ---
CANALES = ["n1fe", "n2cu", "n3zn", "n4mo", "n6sc"]        # los 5 canales de intensidad
METALES = ["n1fe", "n2cu", "n3zn", "n4mo"]                # canales que se ortogonalizan (sin n6sc)
FEATS_CLUSTER = ["n1fe_ortho", "n2cu_ortho",
                 "n3zn_ortho", "n4mo_ortho"]              # features del clustering (4 ortho, sin n6sc)
FEATS_REGRESION = ["n1fe_ortho", "n2cu_ortho",
                   "n3zn_ortho", "n4mo_ortho", "n6sc"]    # features de la regresión (ortho + n6sc)

# --- Leyes químicas a estimar y el modelo elegido para cada una ---
TARGETS = ["pFe", "pCu", "pMo", "pZn"]                    # leyes de laboratorio
MODELO_POR_TARGET = {"pFe": "Ridge", "pCu": "SVR",        # familia de modelo por ley
                     "pMo": "Ridge", "pZn": "PLS"}

# --- Semilla global para reproducibilidad ---
RANDOM_STATE = 42                                         # fija el azar en todo el pipeline

# --- Nombres de los artefactos serializados (una fuente de verdad) ---
ART_ANOMALIAS = "anomaly_detector.joblib"                # etapa 1 (limpieza)
ART_ORTHO = "orthogonalization_regressions.joblib"       # etapa 2 (ortogonalización)
ART_POWER = "power_transformer.joblib"                   # etapa 3 (escalado)
ART_SCALER = "scaler.joblib"                             # etapa 3 (escalado)
ART_KMEANS = "kmeans_model.joblib"                       # etapa 4 (clustering)
ART_GMM = "gmm_model.joblib"                             # etapa 4 (clustering)
ART_REGRESION = "modelos_locales_por_cluster.joblib"     # etapa 5 (modelos locales)

# ============================================================
# RECALIBRACIÓN CONTRA COMPÓSITO DE 12 H  (cambios 1-4)
# ============================================================
# Justificación medida (ver notebooks/10_recalibracion_composito.py):
#   - El compósito de 12 h es la ÚNICA referencia válida de laboratorio: lo corta
#     personal de metalurgia justo antes de la etapa de rayos X, así que es el
#     mismo material que ve la celda (sin proceso intermedio).
#   - Colocación triple (modelo / calibración de fábrica / compósito, n=184) dio
#     el techo de correlación alcanzable por elemento:
#         Cu techo=0.634  fábrica=0.517  modelo=0.505   -> margen 0.13 (poco)
#         Fe techo=0.743  fábrica=0.470  modelo=0.278   -> margen 0.46 (TODO el margen)
#         Mo techo=0.961  fábrica=0.948  modelo=0.958   -> ya en el límite
# ============================================================

# --- Fuente de laboratorio: compósito de 12 h (tags MAN del courier) ---
LAB_COMPOSITO = DATA_RAW / "assay_lab_courier_pi.csv"    # compósito 12 h (turno día 07:30 / noche 19:30)
# Mapeo VERIFICADO contra el descriptor real del PI (2026-09-26), no deducido.
# La version anterior de este diccionario tenia Fe y Cu invertidos y suponia un
# tag de Zn que no existe. El error era solo documental -- ninguna etapa lo
# usaba -- y se confirmo que las columnas de assay_lab_courier_pi.csv estan
# bien nombradas (identicas al PI hasta 1e-6).
TAGS_COMPOSITO = {"7100AIP101MAN": "fe",                 # %Fe Promedio Concentrado colectivo Courier Cobre
                  "7100AIP102MAN": "cu",                 # %Cu Promedio Concentrado colectivo Courier Cobre
                  "7100AIP103MAN": "ins",                # %Ins Concentrado colectivo Courier Cobre
                  "7100AIP104MAN": "mo"}                 # %Moly Promedio Concentrado colectivo Courier Cobre
# NOTA: el composito NO trae Zn. Y SI trae INSOLUBLES, que es la variable que
# en el diagnostico dio +0.144 en pFe al usarse para rutear modelos locales.
# Ver docs/bitacora_analisis.md seccion 2.6 y el hilo abierto 5.4.
COL_TS_COMPOSITO = "Unnamed: 0"                          # columna de timestamp tal como la exporta PI
TZ_COMPOSITO = "America/Lima"                            # PI exporta en UTC-05:00 -> se normaliza a hora local

# --- Alineación intensidades <-> compósito ---
VENTANA_COMPOSITO_H = 12                                 # ancho de la ventana centrada en el ensayo (= cadencia del lab)
MIN_BLOQUES_VENTANA = 8                                  # mínimo de bloques de 30 min para aceptar la ventana
BLOQUE_RESAMPLE = "30min"                                # submuestreo previo: descorrelaciona (autocorr lag-1 = 0.907)

# --- Cambio 2: ventana móvil de reentrenamiento ---
# Barrido medido sobre pFe (corr del backtest walk-forward, n=228):
#   120 d -> 0.465    180 d -> 0.489    270 d -> 0.508
# Con 270 d gana porque la ventana móvil ya elimina la deriva de largo plazo y
# lo que manda pasa a ser el tamaño de muestra. Más allá vuelve a contaminar.
DIAS_VENTANA_MOVIL = 270                                 # ancho de la ventana móvil de entrenamiento
PASO_REENTRENO_DIAS = 15                                 # cada cuántos días se reajusta (15 d: 0.508 vs 30 d: 0.500)

# --- Cambio 3: corrección de sesgo en línea ---
N_MUESTRAS_SESGO = 20                                    # nº de compósitos recientes para estimar el offset
SESGO_POR_TURNO = True                                   # sesgo separado día/noche (medido en Fe: 0.596 vs 0.385)

# --- Leyes a recalibrar contra el compósito y su columna en el lab ---
MAPA_LEY_COMPOSITO = {"pFe": "fe", "pCu": "cu", "pMo": "mo"}  # pZn NO está en el compósito -> conserva el modelo antiguo

# SOLO HIERRO. El backtest walk-forward (n=228) comparado contra la línea base
# medida por colocación triple es inequívoco:
#
#     ley   antes   recalibrado   fábrica   techo
#     pFe   0.278       0.508       0.470   0.743   -> MEJORA y supera a fábrica
#     pCu   0.505       0.389       0.517   0.634   -> EMPEORA, no recalibrar
#     pMo   0.958       0.951       0.948   0.961   -> ya en el techo, no tocar
#
# Cu empeora porque su modelo original se calibra con muestras puntuales
# emparejadas al instante exacto de una lectura: para cobre esa señal es más
# limpia que el promedio de una ventana de 12 h. Mo ya está en el límite de
# información del sistema, así que sólo puede perder.
LEYES_RECALIBRAR = ["pFe"]                               # cambiar solo con evidencia de backtest que lo respalde

# --- Cambio 4: detector de anomalías robusto a la deriva ---
# El instrumento pierde ~22%/año de cuentas (decaimiento de fuente). Entrenado
# sobre intensidades ABSOLUTAS, el IsolationForest rechazaba 16-33% del trimestre
# corriente. Sobre fracciones de cierre (derivan 1.2-1.3%) baja a ~3.5%.
FEATS_ANOMALIA = ["n1fe_f", "n2cu_f", "n3zn_f", "n4mo_f"]  # fracciones de cierre m/Σm (invariantes a ganancia común)
CONTAMINACION_ANOMALIA = 0.02                            # proporción esperada de anomalías reales

# --- Artefactos nuevos ---
ART_CALIB_COMPOSITO = "calibracion_composito.joblib"     # cambios 1-2 (modelos entrenados vs compósito)
ART_CORRECTOR_SESGO = "corrector_sesgo.joblib"           # cambio 3 (offset en línea)
ART_MANIFIESTO = "manifiesto_recalibracion.json"         # reproducibilidad: hashes, parámetros y métricas
