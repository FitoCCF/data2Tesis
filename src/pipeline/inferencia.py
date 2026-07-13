# ============================================================
# src/pipeline/inferencia.py — Etapa 6: estimación de leyes en producción
# ============================================================
# Clase EstimadorHibrido: carga los artefactos congelados y estima las leyes de
# una lectura nueva, ruteando cada ley a su mejor modelo (local o global) según
# la tabla de ruteo. Reemplaza el frágil exec(open(...)) por imports normales.
# ============================================================

from collections import deque                             # buffer para sensor congelado
import warnings                                            # silenciar warning cosmético
import numpy as np                                         # cálculo numérico
import joblib                                              # cargar artefactos .joblib

from .config import (CANALES, METALES, FEATS_CLUSTER, MODELS_DIR,  # constantes y rutas
                     ART_ANOMALIAS, ART_ORTHO, ART_POWER, ART_SCALER,
                     ART_KMEANS, ART_GMM, ART_REGRESION)

# Silencia solo el aviso de sklearn sobre nombres de columna (no afecta resultados)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")


class EstimadorHibrido:
    """Carga los modelos congelados y estima leyes ruteando local/global."""

    def __init__(self, models_dir=MODELS_DIR, buffer_size=10, umbral_congelado=3):
        # --- Cargar cada artefacto una sola vez ---
        self.detector = joblib.load(models_dir / ART_ANOMALIAS)   # IsolationForest de calidad
        self.regresiones_ortho = joblib.load(models_dir / ART_ORTHO)  # dict metal -> regresión vs n6sc
        self.power = joblib.load(models_dir / ART_POWER)          # PowerTransformer
        self.scaler = joblib.load(models_dir / ART_SCALER)        # StandardScaler del clustering
        self.kmeans = joblib.load(models_dir / ART_KMEANS)        # modelo KMeans
        self.gmm = joblib.load(models_dir / ART_GMM)              # modelo GMM

        bundle = joblib.load(models_dir / ART_REGRESION)          # bundle de regresión (celda 5)
        self.modelos_locales = bundle["modelos_locales"]          # (ley, cluster) -> pipeline local
        self.modelos_globales = bundle["modelos_globales"]        # ley -> pipeline global
        self.tabla_ruteo = bundle["tabla_ruteo"]                  # (ley, cluster) -> 'local'|'global'
        self.features_reg = bundle["features"]                    # orden de features de la regresión
        self.targets = bundle["targets"]                          # leyes a estimar
        cluster_col = bundle["cluster_col"]                       # columna de cluster usada

        # Elige el modelo de clustering según lo que usó la celda 5
        self.modelo_cluster = self.gmm if "gmm" in cluster_col else self.kmeans

        self.buffer = deque(maxlen=buffer_size)                  # buffer de últimas lecturas
        self.umbral_congelado = umbral_congelado                 # nº de repeticiones que marca congelado

    # ---------- Compuerta de calidad ----------

    def _error_sensor(self, x):
        return bool(np.any(x == -9999) or np.all(x == 0))        # error duro del analizador

    def _sensor_congelado(self, x):
        self.buffer.append(tuple(x))                            # registra la lectura
        return sum(1 for v in self.buffer if v == tuple(x)) >= self.umbral_congelado  # repetida?

    def _es_anomalia(self, x):
        return self.detector.predict(x.reshape(1, -1))[0] == -1  # IsolationForest

    # ---------- Transformaciones ----------

    def _ortogonalizar(self, ints):
        n6sc = np.array([[ints["n6sc"]]])                       # referencia de dilución
        ortho = {}                                              # residuos
        for metal in METALES:                                   # por cada metal
            ortho[f"{metal}_ortho"] = ints[metal] - self.regresiones_ortho[metal].predict(n6sc)[0]  # residuo
        ortho["n6sc"] = ints["n6sc"]                            # conserva n6sc
        return ortho

    def _asignar_cluster(self, ortho):
        v = np.array([[ortho[c] for c in FEATS_CLUSTER]])       # vector de 4 ortho
        v = self.scaler.transform(self.power.transform(v))      # power -> escala
        return int(self.modelo_cluster.predict(v)[0])           # cluster asignado

    # ---------- Predicción ----------

    def predecir(self, ints: dict) -> dict:
        """ints: dict con n1fe, n2cu, n3zn, n4mo, n6sc (intensidades crudas)."""
        x = np.array([ints[c] for c in CANALES], dtype=float)   # vector crudo ordenado
        r = {"leyes": None, "cluster": None, "ruteo": {}, "alertas": [], "confiable": True}  # salida

        if self._error_sensor(x):                              # 1) error duro
            r["alertas"].append("Error del analizador (-9999) o lectura en cero.")
            r["confiable"] = False
            return r                                           # no estima nada

        if self._sensor_congelado(x):                          # 2) sensor congelado
            r["alertas"].append(f"Lectura repetida >= {self.umbral_congelado} veces: sensor congelado.")
            r["confiable"] = False

        if self._es_anomalia(x):                               # 3) anomalía multivariada
            r["alertas"].append("Lectura marcada como anomalía.")
            r["confiable"] = False

        ortho = self._ortogonalizar(ints)                      # 4) ortogonalización
        cluster = self._asignar_cluster(ortho)                 # 5) asignación de cluster
        r["cluster"] = cluster

        vector = np.array([[ortho[f] for f in self.features_reg]])  # features de regresión
        leyes = {}                                             # leyes estimadas
        for ley in self.targets:                               # por cada ley
            decision = self.tabla_ruteo[(ley, cluster)]        # 'local' o 'global'
            r["ruteo"][ley] = decision                         # registra el ruteo
            modelo = (self.modelos_locales[(ley, cluster)]     # elige el modelo
                      if decision == "local" else self.modelos_globales[ley])
            leyes[ley] = float(np.ravel(modelo.predict(vector))[0])  # predicción
        r["leyes"] = leyes                                     # guarda las leyes
        return r                                               # devuelve el resultado
