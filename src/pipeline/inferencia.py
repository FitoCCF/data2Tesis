# ============================================================
# src/pipeline/inferencia.py — Etapa 6: estimación de leyes en producción
# ============================================================
# Dos estimadores, según cuánto del pipeline se necesite en línea:
#
#   EstimadorCluster  -> solo etapas 1-4 (limpieza, ortogonalización, escalado,
#                         clustering). Útil para "solo saber a qué régimen de
#                         mineral pertenece esta lectura", sin necesitar el
#                         bundle de modelos de la etapa 5.
#   EstimadorHibrido   -> pipeline completo (etapas 1-6): además rutea cada ley
#                         al modelo local o global y devuelve las leyes estimadas.
#
# Ambos cargan los artefactos congelados una sola vez y reemplazan el frágil
# exec(open(...)) de los notebooks originales por imports normales.
#
# CLI (scoring por lotes de un CSV de lecturas nuevas):
#   python -m src.pipeline.inferencia --input data/processed/septiembre.csv --hasta-etapa clustering
#   python -m src.pipeline.inferencia --input data/processed/septiembre.csv --hasta-etapa leyes
# ============================================================

from collections import deque                             # buffer para sensor congelado
import warnings                                            # silenciar warning cosmético
import numpy as np                                         # cálculo numérico
import pandas as pd                                        # DataFrames (features de cierre del detector)
import joblib                                               # cargar artefactos .joblib

from .config import (CANALES, METALES, FEATS_CLUSTER, MODELS_DIR,  # constantes y rutas
                     ART_ANOMALIAS, ART_ORTHO, ART_POWER, ART_SCALER,
                     ART_KMEANS, ART_GMM, ART_REGRESION,
                     ART_CALIB_COMPOSITO, ART_CORRECTOR_SESGO)
from .features import features_cierre                     # fracciones de cierre (detector robusto, cambio 4)

# Silencia solo el aviso de sklearn sobre nombres de columna (no afecta resultados)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")


class _CompuertaYCluster:
    """Mixin con la compuerta de calidad + transformaciones hasta cluster
    (etapas 1-4), compartido por EstimadorCluster y EstimadorHibrido para que
    ambos apliquen EXACTAMENTE la misma lógica."""

    def _cargar_etapas_1_4(self, models_dir, buffer_size, umbral_congelado):
        self.detector = joblib.load(models_dir / ART_ANOMALIAS)   # IsolationForest de calidad (etapa 1)
        self.regresiones_ortho = joblib.load(models_dir / ART_ORTHO)  # dict metal -> regresión vs n6sc (etapa 2)
        self.power = joblib.load(models_dir / ART_POWER)          # PowerTransformer (etapa 3)
        self.scaler = joblib.load(models_dir / ART_SCALER)        # StandardScaler (etapa 3)
        self.kmeans = joblib.load(models_dir / ART_KMEANS)        # modelo KMeans (etapa 4)
        self.gmm = joblib.load(models_dir / ART_GMM)              # modelo GMM (etapa 4)

        self.buffer = deque(maxlen=buffer_size)                  # buffer de últimas lecturas
        self.umbral_congelado = umbral_congelado                 # nº de repeticiones que marca congelado

    def _error_sensor(self, x):
        return bool(np.any(x == -9999) or np.all(x == 0))        # error duro del analizador

    def _sensor_congelado(self, x):
        self.buffer.append(tuple(x))                            # registra la lectura
        return sum(1 for v in self.buffer if v == tuple(x)) >= self.umbral_congelado  # repetida?

    def _es_anomalia(self, x):
        """CAMBIO 4: aplica el detector en el MISMO espacio en que se entrenó.

        Si el artefacto se entrenó sobre fracciones de cierre (marca _robusto),
        hay que transformar la lectura antes de evaluarla; si se entrenó sobre
        intensidades absolutas (artefacto histórico), se evalúa tal cual.
        Aplicar un detector de cierre a intensidades absolutas marcaría TODO
        como anomalía, así que el modo no puede quedar implícito.
        """
        if getattr(self.detector, "_robusto", False):     # detector entrenado sobre fracciones de cierre
            fila = pd.DataFrame([dict(zip(CANALES, x))])  # arma un DataFrame de una fila
            v = features_cierre(fila).values               # lo pasa a fracciones m/Σm
        else:                                             # detector histórico (intensidades absolutas)
            v = x.reshape(1, -1)                          # se evalúa el vector crudo
        return self.detector.predict(v)[0] == -1          # -1 = anomalía

    def _ortogonalizar(self, ints):
        n6sc = np.array([[ints["n6sc"]]])                       # referencia de dilución
        ortho = {}                                              # residuos
        for metal in METALES:                                   # por cada metal
            ortho[f"{metal}_ortho"] = ints[metal] - self.regresiones_ortho[metal].predict(n6sc)[0]  # residuo
        ortho["n6sc"] = ints["n6sc"]                            # conserva n6sc
        return ortho

    def _asignar_cluster(self, ortho, modelo_cluster):
        v = np.array([[ortho[c] for c in FEATS_CLUSTER]])       # vector de 4 ortho
        v = self.scaler.transform(self.power.transform(v))      # power -> escala
        return int(modelo_cluster.predict(v)[0])                # cluster asignado


class EstimadorCluster(_CompuertaYCluster):
    """Solo etapas 1-4: compuerta de calidad + cluster (régimen de mineral).
    No requiere el bundle de la etapa 5 -> útil para poner en línea solo el
    clustering (p.ej. clasificar lecturas nuevas sin (aún) leyes de lab)."""

    def __init__(self, models_dir=MODELS_DIR, usar="gmm", buffer_size=10, umbral_congelado=3):
        self._cargar_etapas_1_4(models_dir, buffer_size, umbral_congelado)
        self.modelo_cluster = self.gmm if usar == "gmm" else self.kmeans

    def predecir(self, ints: dict) -> dict:
        """ints: dict con n1fe, n2cu, n3zn, n4mo, n6sc (intensidades crudas)."""
        x = np.array([ints[c] for c in CANALES], dtype=float)
        r = {"cluster": None, "alertas": [], "confiable": True}

        if self._error_sensor(x):
            r["alertas"].append("Error del analizador (-9999) o lectura en cero.")
            r["confiable"] = False
            return r

        if self._sensor_congelado(x):
            r["alertas"].append(f"Lectura repetida >= {self.umbral_congelado} veces: sensor congelado.")
            r["confiable"] = False

        if self._es_anomalia(x):
            r["alertas"].append("Lectura marcada como anomalía.")
            r["confiable"] = False

        ortho = self._ortogonalizar(ints)
        r["cluster"] = self._asignar_cluster(ortho, self.modelo_cluster)
        return r


class EstimadorHibrido(_CompuertaYCluster):
    """Pipeline completo (etapas 1-6): carga los modelos congelados y estima
    leyes ruteando local/global por cluster."""

    def __init__(self, models_dir=MODELS_DIR, buffer_size=10, umbral_congelado=3,
                 usar_recalibracion=True):
        self._cargar_etapas_1_4(models_dir, buffer_size, umbral_congelado)

        bundle = joblib.load(models_dir / ART_REGRESION)          # bundle de regresión (etapa 5)
        self.modelos_locales = bundle["modelos_locales"]          # (ley, cluster) -> pipeline local
        self.modelos_globales = bundle["modelos_globales"]        # ley -> pipeline global
        self.tabla_ruteo = bundle["tabla_ruteo"]                  # (ley, cluster) -> 'local'|'global'
        self.features_reg = bundle["features"]                    # orden de features de la regresión
        self.targets = bundle["targets"]                          # leyes a estimar
        cluster_col = bundle["cluster_col"]                       # columna de cluster usada

        # Elige el modelo de clustering según lo que usó la etapa 5
        self.modelo_cluster = self.gmm if "gmm" in cluster_col else self.kmeans

        # --- CAMBIOS 1-3: calibración contra compósito + corrector de sesgo ---
        # Solo para las leyes que el backtest demostró que mejoran (hoy: pFe).
        # Las demás (pCu, pMo, pZn) siguen con el modelo de la etapa 5, porque
        # medido empeoran con la recalibración o ya están en su techo.
        self.calib_composito = None                               # bundle recalibrado (None = no disponible)
        self.corrector = None                                     # corrector de sesgo en línea
        if usar_recalibracion:                                    # se puede desactivar para reproducir el pipeline viejo
            ruta_calib = models_dir / ART_CALIB_COMPOSITO         # artefacto de los cambios 1-2
            if ruta_calib.exists():                               # solo si ya se entrenó
                self.calib_composito = joblib.load(ruta_calib)    # carga el bundle recalibrado
            ruta_sesgo = models_dir / ART_CORRECTOR_SESGO         # artefacto del cambio 3
            if ruta_sesgo.exists():                               # solo si ya hay historia de sesgo
                self.corrector = joblib.load(ruta_sesgo)          # carga el corrector

    def predecir(self, ints: dict, turno: str = "dia") -> dict:
        """Estima las leyes de una lectura.

        Parámetros
        ----------
        ints : dict con n1fe, n2cu, n3zn, n4mo, n6sc (intensidades crudas).
        turno : 'dia' o 'noche'. Solo se usa para elegir el sesgo del CAMBIO 3
            (medido: el sesgo de Fe difiere 0.211 entre turnos).
        """
        x = np.array([ints[c] for c in CANALES], dtype=float)   # vector crudo ordenado
        r = {"leyes": None, "cluster": None, "ruteo": {}, "alertas": [], "confiable": True}  # salida

        if self._error_sensor(x):                              # 1) error duro
            r["alertas"].append("Error del analizador (-9999) o lectura en cero.")
            r["confiable"] = False
            return r                                           # no estima nada

        if self._sensor_congelado(x):                          # 2) sensor congelado
            r["alertas"].append(f"Lectura repetida >= {self.umbral_congelado} veces: sensor congelado.")
            r["confiable"] = False

        if self._es_anomalia(x):                                # 3) anomalía multivariada
            r["alertas"].append("Lectura marcada como anomalía.")
            r["confiable"] = False

        ortho = self._ortogonalizar(ints)                       # 4) ortogonalización
        cluster = self._asignar_cluster(ortho, self.modelo_cluster)  # 5) asignación de cluster
        r["cluster"] = cluster

        vector = np.array([[ortho[f] for f in self.features_reg]])  # features de regresión (etapa 5)
        fila = pd.DataFrame([{c: ints[c] for c in CANALES}])     # fila cruda (features del modelo recalibrado)

        # Leyes que van al modelo recalibrado contra compósito (hoy: solo pFe)
        leyes_recal = set(self.calib_composito["modelos"]) if self.calib_composito else set()

        leyes = {}                                              # leyes estimadas
        for ley in self.targets:                                # por cada ley
            if ley in leyes_recal:                              # --- ruta RECALIBRADA (cambios 1-3) ---
                from .calibracion_composito import predecir as _pred_recal  # import local: evita ciclo
                valor = float(_pred_recal(self.calib_composito, fila)[ley].iloc[0])  # predicción cruda
                if self.corrector is not None:                  # si hay historia de sesgo
                    valor = self.corrector.corregir(ley, valor, turno)  # aplica el offset (cambio 3)
                r["ruteo"][ley] = "composito"                   # deja constancia de la ruta usada
            else:                                               # --- ruta ORIGINAL (etapa 5, local/global) ---
                decision = self.tabla_ruteo[(ley, cluster)]     # 'local' o 'global'
                r["ruteo"][ley] = decision                      # registra el ruteo
                modelo = (self.modelos_locales[(ley, cluster)]  # elige el modelo
                          if decision == "local" else self.modelos_globales[ley])
                valor = float(np.ravel(modelo.predict(vector))[0])  # predicción
            leyes[ley] = valor                                  # guarda la ley estimada

        r["leyes"] = leyes                                      # guarda las leyes
        return r


# ============================================================
# CLI — scoring por lotes de un CSV de lecturas nuevas
# ============================================================
def _main():
    import argparse
    import pandas as pd

    from .config import DATA_PROCESSED

    ap = argparse.ArgumentParser(description="Etapa 6: inferencia sobre un CSV de lecturas nuevas")
    ap.add_argument("--input", required=True, help="CSV con columnas n1fe, n2cu, n3zn, n4mo, n6sc")
    ap.add_argument("--output", default=None, help="CSV de salida (default: <input>_scored.csv)")
    ap.add_argument("--hasta-etapa", choices=["clustering", "leyes"], default="leyes",
                    help="'clustering' usa EstimadorCluster (solo etapas 1-4, no requiere la etapa 5); "
                         "'leyes' usa EstimadorHibrido (pipeline completo, requiere la etapa 5)")
    ap.add_argument("--modelo-cluster", choices=["gmm", "kmeans"], default="gmm",
                    help="Solo aplica con --hasta-etapa clustering")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    est = EstimadorCluster(usar=args.modelo_cluster) if args.hasta_etapa == "clustering" else EstimadorHibrido()

    cols_extra = [c for c in ("date", "time", "instance") if c in df.columns]  # se conservan si existen

    # El turno hace falta para la corrección de sesgo del CAMBIO 3 (el sesgo de
    # Fe difiere 0.211 entre día y noche). Se deduce de la hora si el CSV la trae;
    # si no, cae a 'dia' y el corrector usa el sesgo global como respaldo.
    if "time" in df.columns:                                # el CSV trae hora de la lectura
        # Se extrae la hora con regex en vez de pd.to_datetime: las horas vienen
        # como 'HH:MM:SS' y el parseo completo es lento y emite un warning de
        # formato ambiguo sobre decenas de miles de filas.
        horas = (df["time"].astype(str).str.extract(r"^\s*(\d{1,2})")[0]  # primer grupo de 1-2 dígitos
                 .astype("Float64"))                        # nullable: deja NaN si no matcheó
        turnos = horas.map(lambda h: "noche" if pd.notna(h) and h >= 12 else "dia")  # turno A = mañana
    else:                                                   # CSV sin hora
        turnos = pd.Series(["dia"] * len(df), index=df.index)  # respaldo: sesgo global

    filas = []
    for _, fila in df.iterrows():
        ints = {c: fila[c] for c in CANALES}
        res = est.predecir(ints, turno=turnos.loc[fila.name]) \
            if isinstance(est, EstimadorHibrido) else est.predecir(ints)  # EstimadorCluster no usa turno
        salida_fila = {c: fila[c] for c in cols_extra}          # date/time/instance primero, si existen
        salida_fila.update(ints)
        salida_fila["cluster"] = res.get("cluster")
        salida_fila["alertas"] = "; ".join(res.get("alertas") or [])
        salida_fila["confiable"] = res.get("confiable")
        if res.get("leyes"):                                    # aplana pFe/pCu/pMo/pZn como columnas numéricas
            salida_fila.update(res["leyes"])
        if res.get("ruteo"):                                     # ruteo aparte, sin pisar las leyes (mismo nombre)
            salida_fila.update({f"ruteo_{k}": v for k, v in res["ruteo"].items()})
        filas.append(salida_fila)

    salida = pd.DataFrame(filas)
    out_path = args.output or str(args.input).rsplit(".", 1)[0] + "_scored.csv"
    salida.to_csv(out_path, index=False)
    print(f"Filas procesadas: {len(salida)}  |  confiables: {int(salida['confiable'].sum())}")
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    _main()
