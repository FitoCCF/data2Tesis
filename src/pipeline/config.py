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

# --- Base de datos Postgres: tabla y columnas extraíbles ---
# Única fuente de verdad de qué se puede pedir a la BD. src/database/extractor.py
# arma la consulta a partir de estos grupos: agregar una ley o un canal nuevo
# es agregar un nombre aquí, no editar SQL.
TABLA_BD = "works4cdp_assay"                              # tabla que llena api2db.py (data4cdpv1_local)
SAMPLE_ID_COURIER = 24                                    # "Concentrado Colectivo" en works4cdp_sample
LLAVES_BD = ["date", "time", "instance"]                  # siempre se extraen: identifican la fila
COLS_BD = {
    "intensidades": ["n1fe", "n2cu", "n3zn", "n4mo",      # canales del analizador (courier)
                     "n5ech5", "n6sc", "n7ech7"],
    "leyes": ["pFe", "pCu", "pZn", "pMo", "pIns", "pSol"],  # ensayos de laboratorio (misma fila)
}

# --- PI OSIsoft vía pasarela PiGateway (src/acquisition/pi_client.py) ---
# El token NO va en el código (el repo está en GitHub): se lee de PI_TOKEN o
# de ~/.pi_token (ver src/acquisition/from_pi.py: cargar_token).
PI_HOST = "10.25.18.85"                                   # pasarela; PI_GATEWAY_HOST la sobreescribe
PI_PUERTO = 5173                                          # puerto de esta instalación (no el 5000 del cliente)
FECHA_INICIO_COURIER = "2025-07-13 15:00:00"              # no hay historia de los tags del courier antes
TAGS_COURIER = {                                          # tag PI -> canal del pipeline
    "_296290_ConcFinal_CanalFe_ABB": "n1fe",
    "_296290_ConcFinal_CanalCu_ABB": "n2cu",
    "_296290_ConcFinal_CanalZn_ABB": "n3zn",
    "_296290_ConcFinal_CanalMo_ABB": "n4mo",
    "_296290_ConcFinal_CanalSc_ABB": "n6sc",
}
# El PI no trae n5ech5 ni n7ech7: solo la BD los tiene.

# --- Crudos de la etapa 0: UN archivo por fuente, no se modifican después ---
#   A  muestreo con ensayo de laboratorio (BD, ~cada 3 días, desde 2022-12)
#   B  operación continua del courier (PI recorded: eventos cada 100 s, lecturas reales ~cada 21 min, desde 2025-07-13)
#   C  compósito de laboratorio de 12 h (PI, fe/cu/ins/mo; sin Zn ni Sol)
RAW_COURIER_BD = DATA_RAW / "courier_bd.csv"
RAW_COURIER_PI = DATA_RAW / "courier_pi.csv"
RAW_COMPOSITO = DATA_RAW / "composito_pi.csv"

# --- Etapa 1a: limpieza por fuente (src/pipeline/limpieza_fuentes.py) ---
LIMPIO_COURIER_PI = DATA_PROCESSED / "courier_pi_lecturas.csv"   # una fila por lectura real del courier
LIMPIO_COURIER_BD = DATA_PROCESSED / "courier_bd_limpio.csv"
LIMPIO_COMPOSITO = DATA_PROCESSED / "composito_limpio.csv"
TOL_EVENTO = "5s"             # eventos de metales a <= esto son el mismo ciclo (a veces n1fe llega 1 s antes)
TOL_N6SC = "60s"              # ventana para pegar n6sc a la lectura de los metales (solo actúa en filas partidas)
UMBRAL_SOSTENIDA = "2h"       # lectura republicada más de esto -> bandera 'sostenida' (p99 normal: 33 min)
Z_LEY_SOSPECHOSA = 5.0        # |ley - mediana| / (1.4826·MAD) por encima -> candidata a bandera, no borrado
Z_CANAL_RESPALDO = 2.0        # ...salvo que su canal del courier también se desvíe > esto en la misma dirección
CANAL_DE_LEY = {"pFe": "n1fe", "pCu": "n2cu", "pZn": "n3zn", "pMo": "n4mo"}  # pIns/pSol sin canal propio

# --- Etapa 1b: unificación A + B (src/pipeline/unificacion.py) ---
UNIFICADO = DATA_PROCESSED / "courier_unificado.csv"     # tabla maestra: una fila por lectura del courier
VENTANA_CALCE = "13h"         # busca la lectura del PI con los mismos valores a <= esto de la hora de la BD
                              # (13 h y no minutos: 3 filas de la BD de jun-2026 tienen la hora corrida 12 h)
UMBRAL_HORA_BD = "1h"         # calce más lejos que esto -> bandera 'hora_bd_desfasada' (medido: p95 = 28 min)

# --- Corte único del pipeline (etapa 1c en adelante) ---
# Toda etapa que AJUSTA algo (IsolationForest, escalado,
# GMM, modelos) usa solo lecturas hasta este día inclusive. Lo posterior es la
# prueba final. 2026-05-31: train 347 muestras con ley; test jun-ago con 176
# compósitos y 18 muestras de la BD; deja en train la mayor parte del período
# con n6sc desfasado (abr-jun 2026). Ver docs/bitacora_analisis.md.
FECHA_CORTE_TRAIN = "2026-05-31"
FECHA_INICIO_PRODUCCION = "2026-09-01"                   # desde aquí periodo = "produccion" (datos nuevos, nunca vistos)
LIMPIO = DATA_PROCESSED / "courier_limpio.csv"           # tabla maestra con banderas de la 1c

# --- Canales espectrales ---
# n6sc es el canal de dispersión (Sc). NO mide el sólido de la muestra:
# corr(n6sc, pSol) = -0.15 sobre 344 muestras; log(ΣI) sí (+0.57).
CANALES = ["n1fe", "n2cu", "n3zn", "n4mo", "n6sc"]        # los 5 canales de intensidad
METALES = ["n1fe", "n2cu", "n3zn", "n4mo"]                # canales de metal (sin n6sc)

# --- Etapa 2: features robustas a la deriva (src/pipeline/features.py) ---
# Reemplaza a la ortogonalización contra n6sc (2026-09-30): sus residuos
# arrastraban la deriva del instrumento (n2cu_ortho de +6,600 en 2022H2 a
# -5,100 en 2026H2 con pCu casi constante), así que un GMM sobre ellos
# separaría épocas en vez de mineral. Transformación fija: no se ajusta nada.
FEATS_CLUSTER = ["lr_fe_cu", "lr_zn_cu", "lr_mo_cu"]      # log-cocientes contra Cu (alr, Aitchison)
FEATS_REGRESION = ["n1fe_f", "n2cu_f", "n3zn_f", "n4mo_f", "logSumI"]  # fracciones de cierre + magnitud
FEATURES = DATA_PROCESSED / "courier_features.csv"       # salida de la etapa 2
# Corrección de dilución (2026-09-30). El cociente NO cancela del todo el agua:
# a química fija, pSol p10->p90 mueve lr_fe_cu +0.91 sd (t=+12) y lr_mo_cu
# -0.37 sd (el agua atenúa distinto Fe 6.4 keV y Cu 8.0 keV). n6sc no sirve
# para corregirlo (corr con pSol -0.15). Proxy: dil = logSumI - EWMA(logSumI)
# solo con lecturas pasadas; la EWMA absorbe la deriva lenta de la fuente y
# deja la variación rápida. corr(dil, pSol) = +0.55 con halflife 90 d (7 d: 0.50).
DIL_HALFLIFE = "90D"
# Solo se corrige donde el laboratorio demostró el efecto (a química fija):
# lr_fe_cu t=+12 -> se corrige; lr_mo_cu pendiente ~0 (no cambia nada);
# lr_zn_cu SIN efecto (t=-1.3) y la corrección le CREABA uno (t=+3.3) -> no.
DIL_CORREGIR = ["lr_fe_cu"]
ESCALADO = DATA_PROCESSED / "courier_escalado.csv"       # salida de la etapa 3 (+ columnas <feature>_z)
CLUSTERIZADO = DATA_PROCESSED / "courier_clusterizado.csv"  # salida de la etapa 4
K_GRUPOS = 4                  # grupos del GMM (etapa 4). Elegido por diagnostico_k(): silhouette 0.249 y
                              # ARI 0.856, máximo local para k>=3, y cada grupo sobre-representa un elemento
                              # distinto confirmado por laboratorio. k=2 es la alternativa más estable (ARI 0.96).

# --- Leyes químicas a estimar y el modelo elegido para cada una ---
TARGETS = ["pFe", "pCu", "pMo", "pZn"]                    # leyes de laboratorio
MODELO_POR_TARGET = {"pFe": "Ridge", "pCu": "SVR",        # familia de modelo por ley
                     "pMo": "Ridge", "pZn": "PLS"}

# --- Semilla global para reproducibilidad ---
RANDOM_STATE = 42                                         # fija el azar en todo el pipeline

# --- Nombres de los artefactos serializados (una fuente de verdad) ---
ART_ANOMALIAS = "anomaly_detector.joblib"                # etapa 1 (limpieza)
ART_SCALER = "scaler.joblib"                             # etapa 3 (escalado)
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
LAB_COMPOSITO = RAW_COMPOSITO                            # compósito 12 h (turno día 07:30 / noche 19:30), etapa 0
# Mapeo VERIFICADO contra el descriptor real del PI (2026-09-26), no deducido.
# La version anterior de este diccionario tenia Fe y Cu invertidos y suponia un
# tag de Zn que no existe. El error era solo documental -- ninguna etapa lo
# usaba -- y se confirmo contra el PI (identicas hasta 1e-6). La etapa 0
# (from_pi.extraer_composito) ya escribe el crudo con estos nombres.
TAGS_COMPOSITO = {"7100AIP101MAN": "fe",                 # %Fe Promedio Concentrado colectivo Courier Cobre
                  "7100AIP102MAN": "cu",                 # %Cu Promedio Concentrado colectivo Courier Cobre
                  "7100AIP103MAN": "ins",                # %Ins Concentrado colectivo Courier Cobre
                  "7100AIP104MAN": "mo"}                 # %Moly Promedio Concentrado colectivo Courier Cobre
# NOTA: el composito NO trae Zn. Y SI trae INSOLUBLES, que es la variable que
# en el diagnostico dio +0.144 en pFe al usarse para rutear modelos locales.
# Ver docs/bitacora_analisis.md seccion 2.6 y el hilo abierto 5.4.
COL_TS_COMPOSITO = "ts"                                  # hora local naive, tal como la escribe la etapa 0

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
