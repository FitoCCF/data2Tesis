# ============================================================
# CELDA 6 — INFERENCIA HÍBRIDA (estimación de leyes en producción)
# ============================================================
# Dada una lectura nueva de intensidades de campo, encadena TODOS los modelos
# guardados por las celdas 1-5 y devuelve las leyes estimadas, ruteando cada
# ley a su mejor modelo (local del cluster o global) según la tabla de ruteo.
#
# Orden del flujo (todo con modelos CONGELADOS, sin recalcular nada):
#   lectura cruda -> compuerta de calidad -> ortogonalización -> escala
#                 -> asignación de cluster -> ruteo por ley -> leyes estimadas
# ============================================================

from collections import deque                          # buffer para detectar sensor congelado
import numpy as np                                      # cálculo numérico
import joblib                                            # cargar modelos serializados

# --- Silenciar SOLO la advertencia cosmética de sklearn sobre nombres de columna ---
# (algunos transformadores se ajustaron con DataFrame y otros con array; la mezcla
#  genera un UserWarning inofensivo que NO afecta los resultados numéricos)
import warnings                                             # manejo de advertencias
from sklearn.exceptions import InconsistentVersionWarning  # (import defensivo)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='X has feature names')

# --- Rutas de los modelos guardados por las celdas anteriores ---
RUTA_ANOMALIAS   = 'models/anomaly_detector.joblib'               # celda 1
RUTA_ORTHO       = 'models/orthogonalization_regressions.joblib'  # celda 2
RUTA_POWER       = 'models/power_transformer.joblib'              # celda 3
RUTA_SCALER      = 'models/scaler.joblib'                         # celda 3
RUTA_KMEANS      = 'models/kmeans_model.joblib'                   # celda 4
RUTA_GMM         = 'models/gmm_model.joblib'                      # celda 4
RUTA_REGRESION   = 'models/modelos_locales_por_cluster.joblib'    # celda 5 (incluye tabla_ruteo)

# --- Orden de canales que espera todo el pipeline ---
CANALES = ['n1fe', 'n2cu', 'n3zn', 'n4mo', 'n6sc']      # n6sc = sólido en la muestra
METALES = ['n1fe', 'n2cu', 'n3zn', 'n4mo']              # canales que se ortogonalizan (todos menos n6sc)
FEATS_CLUSTER = ['n1fe_ortho', 'n2cu_ortho', 'n3zn_ortho', 'n4mo_ortho']  # features del clustering (sin n6sc)


class EstimadorHibrido:
    """Carga todos los modelos y estima leyes ruteando local/global por cluster."""

    def __init__(self, buffer_size=10, umbral_congelado=3):
        # --- Cargar cada modelo una sola vez al instanciar ---
        self.detector_anomalias = joblib.load(RUTA_ANOMALIAS)     # IsolationForest de calidad
        self.regresiones_ortho = joblib.load(RUTA_ORTHO)          # dict: metal -> LinearRegression(vs n6sc)
        self.power = joblib.load(RUTA_POWER)                      # PowerTransformer (Yeo-Johnson)
        self.scaler = joblib.load(RUTA_SCALER)                    # StandardScaler del clustering
        self.kmeans = joblib.load(RUTA_KMEANS)                    # modelo KMeans
        self.gmm = joblib.load(RUTA_GMM)                          # modelo GMM

        bundle = joblib.load(RUTA_REGRESION)                      # bundle de regresión (celda 5)
        self.modelos_locales = bundle['modelos_locales']         # (ley, cluster) -> pipeline local
        self.modelos_globales = bundle['modelos_globales']       # ley -> pipeline global
        self.tabla_ruteo = bundle['tabla_ruteo']                 # (ley, cluster) -> 'local' | 'global'
        self.features_reg = bundle['features']                   # orden de features de la regresión
        self.targets = bundle['targets']                         # leyes a estimar
        self.cluster_col = bundle['cluster_col']                 # 'cluster_gmm' o 'cluster_kmeans'

        # Elige el modelo de clustering según lo que usó la celda 5 para segmentar
        self.modelo_cluster = self.gmm if self.cluster_col == 'cluster_gmm' else self.kmeans

        # Buffer para detectar sensor congelado (patrón temporal, no de una sola lectura)
        self.buffer = deque(maxlen=buffer_size)                  # guarda las últimas lecturas
        self.umbral_congelado = umbral_congelado                 # nº de repeticiones que marca "congelado"

    # ---------- Compuerta de calidad ----------

    def _error_sensor(self, x):
        # True si hay código de error (-9999) o lectura en cero total
        return bool(np.any(x == -9999) or np.all(x == 0))

    def _sensor_congelado(self, x):
        # True si este vector exacto ya apareció >= umbral veces en el buffer reciente
        self.buffer.append(tuple(x))                            # registra la lectura actual
        return sum(1 for v in self.buffer if v == tuple(x)) >= self.umbral_congelado

    def _es_anomalia(self, x):
        # Aplica el IsolationForest de la celda 1 (espera los 5 canales crudos)
        return self.detector_anomalias.predict(x.reshape(1, -1))[0] == -1

    # ---------- Transformaciones ----------

    def _ortogonalizar(self, ints):
        # Descuenta de cada metal la componente explicada por n6sc (dilución/sólido)
        n6sc = np.array([[ints['n6sc']]])                       # referencia como matriz 2D
        ortho = {}                                              # dict de residuos
        for metal in METALES:                                   # por cada metal
            pred = self.regresiones_ortho[metal].predict(n6sc)[0]  # parte explicada por n6sc
            ortho[f'{metal}_ortho'] = ints[metal] - pred         # residuo = señal sin dilución
        ortho['n6sc'] = ints['n6sc']                            # conserva n6sc (lo usa la regresión)
        return ortho

    def _asignar_cluster(self, ortho):
        # Construye el vector de clustering (4 ortho, sin n6sc), lo transforma y predice cluster
        v = np.array([[ortho[c] for c in FEATS_CLUSTER]])       # vector 2D con las 4 ortho
        v = self.power.transform(v)                             # Yeo-Johnson (celda 3)
        v = self.scaler.transform(v)                            # StandardScaler (celda 3)
        return int(self.modelo_cluster.predict(v)[0])           # etiqueta de cluster

    # ---------- Predicción principal ----------

    def predecir(self, ints):
        """ints: dict con n1fe, n2cu, n3zn, n4mo, n6sc (intensidades crudas de campo)."""
        x = np.array([ints[c] for c in CANALES], dtype=float)   # vector crudo en el orden esperado

        r = {'leyes': None, 'cluster': None, 'ruteo': {}, 'alertas': [], 'confiable': True}  # salida

        # 1) Errores duros -> no se estima nada
        if self._error_sensor(x):
            r['alertas'].append('Error del analizador (-9999) o lectura en cero.')
            r['confiable'] = False
            return r

        # 2) Sensor congelado (patrón temporal)
        if self._sensor_congelado(x):
            r['alertas'].append(f'Lectura repetida >= {self.umbral_congelado} veces: posible sensor congelado.')
            r['confiable'] = False

        # 3) Anomalía multivariada
        if self._es_anomalia(x):
            r['alertas'].append('Lectura marcada como anomalía por el detector.')
            r['confiable'] = False

        # 4) Ortogonalización (descuenta dilución vía n6sc)
        ortho = self._ortogonalizar(ints)

        # 5) Asignación de cluster (régimen / tipo de mineral)
        cluster = self._asignar_cluster(ortho)
        r['cluster'] = cluster

        # 6) Ruteo por ley: cada ley usa el modelo que la TABLA ya decidió
        vector_reg = np.array([[ortho[f] for f in self.features_reg]])  # features de regresión (ortho + n6sc)
        leyes = {}                                              # leyes estimadas
        for ley in self.targets:                                # por cada ley química
            decision = self.tabla_ruteo[(ley, cluster)]         # 'local' o 'global' (congelado)
            r['ruteo'][ley] = decision                          # registra qué modelo se usó
            if decision == 'local':
                modelo = self.modelos_locales[(ley, cluster)]   # modelo local de ese cluster
            else:
                modelo = self.modelos_globales[ley]             # modelo global de respaldo
            leyes[ley] = float(np.ravel(modelo.predict(vector_reg))[0])  # predicción de la ley
        r['leyes'] = leyes
        return r


# ============================================================
# DEMO: simular lecturas de planta
# ============================================================
if __name__ == '__main__':
    import pandas as pd

    est = EstimadorHibrido()                                    # carga todos los modelos

    print('Tabla de ruteo cargada (local vs global):')
    for cl in sorted({cl for (_, cl) in est.tabla_ruteo}):      # por cada cluster presente
        print(f'  cluster {cl}:', {ley: est.tabla_ruteo[(ley, cl)] for ley in est.targets})

    print('\nSimulación de lecturas de campo:')
    stream = pd.read_csv('data/processed/intensidades_cobre.csv').head(500)  # muestras reales
    mostradas = 0
    for _, fila in stream.iterrows():                           # recorre lecturas
        ints = {c: fila[c] for c in CANALES}                    # arma el dict de intensidades
        res = est.predecir(ints)                                # estima
        if res['leyes'] and not res['alertas'] and mostradas < 4:
            le = res['leyes']
            print(f"\n  Lectura {ints}")
            print(f"  -> cluster {res['cluster']}  |  ruteo {res['ruteo']}")
            print(f"  -> pFe={le['pFe']:.2f}  pCu={le['pCu']:.2f}  pMo={le['pMo']:.2f}  pZn={le['pZn']:.3f}")
            mostradas += 1
        if mostradas >= 4:
            break
