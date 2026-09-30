#!/usr/bin/env python3
# ============================================================
# src/pipeline/calibracion_composito.py — Recalibración contra el compósito 12 h
# ============================================================
# Implementa los CAMBIOS 1, 2 y 3 del plan de mejora:
#
#   CAMBIO 1 — Entrenar con el compósito de 12 h (n≈2590) en vez de las muestras
#              puntuales de works4cdp_assay (n=314).
#              Motivo: el compósito lo corta personal de metalurgia justo antes
#              de la etapa de rayos X, así que es EL MISMO MATERIAL que ve la
#              celda, y es la única referencia válida de comparación. Además son
#              8 veces más datos.
#
#   CAMBIO 2 — Entrenar con ventana móvil de 180 días en vez de agrupar 4 años.
#              Motivo: la relación intensidad↔ley NO es estacionaria. Medido,
#              la correlación intra-trimestre (0.33-0.51 en Cu) es MAYOR que la
#              agrupada de 4 años (0.328); en la razón Fe/Cu la agrupada (-0.207)
#              es peor que CUALQUIER trimestre individual (-0.22 a -0.50). Juntar
#              todo el histórico mete la deriva como variable confusora.
#
#   CAMBIO 3 — Corrección de sesgo en línea con los últimos N compósitos.
#              Motivo: el sesgo contra el compósito deambula ±1 punto porcentual
#              entre trimestres, incluso en la calibración de FÁBRICA:
#                  per     sesgo_cu  sesgo_fe
#                  2024Q3     0.462     1.013
#                  2025Q2     1.109     0.499
#                  2025Q4     0.055    -0.051
#                  2026Q1     0.018     0.795
#              Ningún calibrado fijo sobrevive a eso.
#
# Ejecutable de forma independiente:
#   python -m src.pipeline.calibracion_composito --backtest
#   python -m src.pipeline.calibracion_composito --entrenar-final
# ============================================================

from collections import deque                             # buffer de residuos recientes (corrector de sesgo)

import numpy as np                                        # cálculo numérico
import pandas as pd                                       # DataFrames
from sklearn.linear_model import RidgeCV                  # regresión regularizada con alpha por CV interna
from sklearn.preprocessing import StandardScaler          # estandarización interna
from sklearn.pipeline import Pipeline                     # encadena escala + modelo
from sklearn.metrics import r2_score, mean_absolute_error  # métricas

from .config import (CANALES, RANDOM_STATE,               # constantes centralizadas
                     LAB_COMPOSITO, COL_TS_COMPOSITO,
                     VENTANA_COMPOSITO_H, MIN_BLOQUES_VENTANA, BLOQUE_RESAMPLE,
                     DIAS_VENTANA_MOVIL, PASO_REENTRENO_DIAS,
                     N_MUESTRAS_SESGO, SESGO_POR_TURNO,
                     MAPA_LEY_COMPOSITO, LEYES_RECALIBRAR)
from .features import features_regresion, columnas_regresion  # features invariantes a la deriva


# ============================================================
# 1. CARGA Y ALINEACIÓN  (CAMBIO 1)
# ============================================================

def cargar_composito(ruta=LAB_COMPOSITO) -> pd.DataFrame:
    """Carga el compósito de 12 h (data/raw/composito_pi.csv, etapa 0).

    El crudo trae 'ts' en hora local naive (la etapa 0 ya convirtió desde
    UTC), comparable directamente con los timestamps de las intensidades.

    Retorna
    -------
    DataFrame con columnas: ts (naive, hora local) + las leyes disponibles.
    """
    lab = pd.read_csv(ruta)                               # lee el crudo de la etapa 0
    lab = lab.rename(columns={COL_TS_COMPOSITO: "ts"})    # columna de timestamp -> 'ts'

    lab["ts"] = pd.to_datetime(lab["ts"])                 # ya viene en hora local naive (etapa 0)

    leyes = [c for c in MAPA_LEY_COMPOSITO.values() if c in lab.columns]  # leyes realmente presentes
    lab = lab[(lab[leyes] > 0).all(axis=1)]               # descarta filas con ceros (dato inválido del lab)

    return lab.sort_values("ts").reset_index(drop=True)   # ordenado por tiempo, índice limpio


def cargar_intensidades(ruta) -> pd.DataFrame:
    """Carga las intensidades del analizador y las deja indexadas por timestamp.

    Aplica solo los filtros DETERMINISTAS de calidad (códigos de error y ceros).
    No aplica el IsolationForest: la detección de anomalías es de la etapa 1 y
    aquí se quiere el stream completo para promediar sobre la ventana.
    """
    df = pd.read_csv(ruta)                                # lee el CSV de intensidades
    df["ts"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))  # arma el timestamp

    df = df.dropna(subset=CANALES)                        # descarta filas con canales faltantes
    df = df[~(df[CANALES] == -9999).any(axis=1)]          # descarta códigos de error del analizador
    df = df[~(df[CANALES] == 0).any(axis=1)]              # descarta canales en cero (analizador detenido)

    return df.sort_values("ts").set_index("ts")[CANALES]  # indexado por tiempo, solo los 5 canales


def alinear_con_composito(inten: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """Empareja cada ensayo del compósito con la media de intensidades de su ventana.

    Para cada ensayo de laboratorio a la hora t, promedia las intensidades en la
    ventana CENTRADA [t-6h, t+6h]. El centrado (lag 0) no es arbitrario: se
    escaneó el lag de -12h a +12h y el óptimo resultó lag 0 (r_cu 0.345 en lag 0
    contra 0.213 en lag -12h), consistente con que el compósito integra las 12 h
    alrededor del corte.

    Antes de promediar se submuestrea a bloques de 30 min. Motivo: la salida del
    analizador tiene autocorrelación 0.907 a lag-1, así que promediar las lecturas
    crudas infla el n efectivo y subestima el error de la media de ventana.

    Retorna
    -------
    DataFrame con una fila por ensayo: ts, turno, n_bloques, los 5 canales
    promediados, y las leyes de laboratorio.
    """
    media_h = VENTANA_COMPOSITO_H / 2                     # medio ancho de la ventana (6 h)
    bloques = inten.resample(BLOQUE_RESAMPLE).mean().dropna()  # submuestreo a 30 min (descorrelaciona)

    # "ins" no es una ley a recalibrar (no está en MAPA_LEY_COMPOSITO) pero se
    # transporta igual si está presente: la usa src.pipeline.ruteo_insoluble
    # para el experimento de ruteo por régimen (hilo abierto 5.4). No afecta
    # el filtrado de filas de cargar_composito(), solo qué columnas viajan.
    leyes = [c for c in list(MAPA_LEY_COMPOSITO.values()) + ["ins"] if c in lab.columns]  # leyes disponibles

    filas = []                                            # acumulador de filas alineadas
    for _, ensayo in lab.iterrows():                      # recorre cada ensayo del compósito
        t = ensayo["ts"]                                  # instante del corte de la muestra
        ventana = bloques.loc[t - pd.Timedelta(hours=media_h):  # bloques dentro de la ventana centrada
                              t + pd.Timedelta(hours=media_h)]

        if len(ventana) < MIN_BLOQUES_VENTANA:            # ventana con muy pocos bloques
            continue                                      # se descarta (media poco confiable)

        fila = {"ts": t,                                  # timestamp del ensayo
                "turno": "dia" if t.hour < 12 else "noche",  # turno A (07:30) vs turno noche (19:30)
                "n_bloques": len(ventana)}                # nº de bloques promediados (trazabilidad)
        fila.update(ventana.mean().to_dict())             # los 5 canales promediados en la ventana
        fila.update({c: ensayo[c] for c in leyes})        # las leyes de laboratorio del compósito
        filas.append(fila)                                # guarda la fila

    D = pd.DataFrame(filas)                               # dataset supervisado
    D = D.replace([np.inf, -np.inf], np.nan).dropna()     # limpia infinitos y faltantes
    return D.sort_values("ts").reset_index(drop=True)     # ordenado por tiempo (crítico para el split temporal)


# ============================================================
# 2. CORRECTOR DE SESGO EN LÍNEA  (CAMBIO 3)
# ============================================================

class CorrectorSesgo:
    """Mantiene el offset entre lo que predice el modelo y el compósito real.

    Funciona como el ajuste de puntería de un instrumento: si en los últimos N
    compósitos el modelo vino disparando 0.8 puntos alto, se le resta 0.8.

    Se estima por turno cuando SESGO_POR_TURNO=True, porque el sesgo día/noche
    medido NO es igual (en Fe: 0.596 de día contra 0.385 de noche, diferencia
    0.211). Si un turno todavía no tiene suficientes muestras, cae al sesgo
    global para no quedarse sin corrección.
    """

    def __init__(self, n_muestras=N_MUESTRAS_SESGO, por_turno=SESGO_POR_TURNO):
        self.n_muestras = n_muestras                      # cuántos residuos recientes se promedian
        self.por_turno = por_turno                        # estimar sesgo separado día/noche
        self.residuos = {}                                # (ley, turno) -> deque de residuos
        self.residuos_global = {}                         # ley -> deque de residuos (respaldo)

    def _buffer(self, almacen, clave):
        """Devuelve el deque de una clave, creándolo si es la primera vez."""
        if clave not in almacen:                          # primera observación de esa clave
            almacen[clave] = deque(maxlen=self.n_muestras)  # buffer circular de tamaño fijo
        return almacen[clave]                             # deque listo para usar

    def actualizar(self, ley: str, prediccion: float, real: float, turno: str = "dia", ts=None):
        """Registra un par (predicción, valor real de laboratorio) ya conocido.

        Se llama cuando llega un resultado nuevo del compósito. El residuo se
        guarda con signo 'predicho - real': positivo = el modelo sobreestima.

        `ts` se acepta y se ignora -- existe solo para que esta clase tenga la
        misma firma que CorrectorKalman (src/pipeline/corrector_kalman.py),
        que sí lo necesita para escalar el ruido de proceso por tiempo
        transcurrido. Permite intercambiar ambos correctores en backtest().
        """
        residuo = float(prediccion) - float(real)         # residuo con signo
        self._buffer(self.residuos, (ley, turno)).append(residuo)   # buffer del turno
        self._buffer(self.residuos_global, ley).append(residuo)     # buffer global (respaldo)

    def sesgo(self, ley: str, turno: str = "dia") -> float:
        """Sesgo estimado para una ley (y turno). 0.0 si aún no hay datos."""
        if self.por_turno:                                # modo por turno
            buf = self.residuos.get((ley, turno))         # busca el buffer del turno
            if buf and len(buf) >= 3:                     # exige al menos 3 muestras para ser estable
                return float(np.mean(buf))                # sesgo del turno

        buf = self.residuos_global.get(ley)               # respaldo: sesgo global de la ley
        return float(np.mean(buf)) if buf else 0.0        # 0.0 si todavía no hay historia

    def corregir(self, ley: str, prediccion: float, turno: str = "dia") -> float:
        """Aplica la corrección: predicción - sesgo estimado."""
        return float(prediccion) - self.sesgo(ley, turno)  # resta el offset aprendido


# ============================================================
# 3. ENTRENAMIENTO CON VENTANA MÓVIL  (CAMBIOS 1 + 2)
# ============================================================

def _construir_regresor() -> Pipeline:
    """Regresor estándar del pipeline recalibrado: estandarizar + Ridge con alpha por CV.

    Se usa RidgeCV en vez del Ridge(alpha=1.0) fijo del pipeline original: con
    ventana móvil el tamaño del set cambia en cada reentreno, así que el nivel
    de regularización óptimo cambia con él y no puede quedar fijo.

    No se usa red neuronal a propósito. Medido con split temporal sobre estos
    mismos datos, los MLP degradaron el desempeño de forma severa:
        Cu: Ridge R2=0.139  |  MLP(16,) R2=-2.463  |  MLP(128,64,32) R2=-3.050
        Fe: Ridge R2=0.161  |  MLP(64,32) R2=-5.319
    El cuello de botella es informacional, no de capacidad del modelo.
    """
    return Pipeline([                                     # pipeline de dos pasos
        ("escala", StandardScaler()),                     # estandariza las features
        ("modelo", RidgeCV(alphas=np.logspace(-3, 4, 40)))  # Ridge con alpha elegido por CV interna
    ])


def ventana_de_entrenamiento(D: pd.DataFrame, fecha_ref: pd.Timestamp,
                             dias=DIAS_VENTANA_MOVIL) -> pd.DataFrame:
    """Recorta el dataset a los últimos `dias` ANTES de fecha_ref (CAMBIO 2).

    El corte es estrictamente anterior a fecha_ref: nunca se entrena con el punto
    que se va a predecir ni con nada posterior. Eso es lo que hace honesto al
    backtest.
    """
    desde = fecha_ref - pd.Timedelta(days=dias)           # inicio de la ventana móvil
    return D[(D["ts"] >= desde) & (D["ts"] < fecha_ref)]  # ventana [desde, fecha_ref)


def entrenar(D: pd.DataFrame, leyes=LEYES_RECALIBRAR) -> dict:
    """Entrena un regresor por ley sobre el DataFrame que se le pasa.

    No hace ventana móvil por sí mismo: recibe ya el subconjunto que corresponda
    (lo arma ventana_de_entrenamiento). Así la misma función sirve para el
    backtest y para el entrenamiento final.

    Retorna
    -------
    dict con los modelos, el orden de features y metadatos de trazabilidad.
    """
    X = features_regresion(D)[columnas_regresion()]       # features en orden fijo y congelado

    modelos = {}                                          # ley -> pipeline entrenado
    for ley in leyes:                                     # por cada ley a recalibrar
        col_lab = MAPA_LEY_COMPOSITO[ley]                 # columna correspondiente en el compósito
        if col_lab not in D.columns:                      # la ley no está en este compósito
            continue                                      # se salta (p.ej. pZn)
        modelos[ley] = _construir_regresor().fit(X.values, D[col_lab].values)  # ajusta el regresor

    return {                                              # bundle de calibración
        "modelos": modelos,                               # ley -> pipeline
        "features": columnas_regresion(),                 # orden de features esperado en inferencia
        "leyes": list(modelos.keys()),                    # leyes efectivamente recalibradas
        "mapa_ley_lab": MAPA_LEY_COMPOSITO,               # equivalencia ley <-> columna de laboratorio
        "n_entrenamiento": len(D),                        # tamaño del set (trazabilidad)
        "ts_min": str(D["ts"].min()),                     # inicio de la ventana de entrenamiento
        "ts_max": str(D["ts"].max()),                     # fin de la ventana de entrenamiento
        "dias_ventana": DIAS_VENTANA_MOVIL,               # parámetro usado
        "random_state": RANDOM_STATE,                     # semilla (reproducibilidad)
    }


def predecir(bundle: dict, df_intensidades: pd.DataFrame) -> pd.DataFrame:
    """Aplica un bundle de calibración a intensidades ya promediadas por ventana.

    df_intensidades debe traer los canales crudos (n1fe..n6sc); las features se
    recalculan aquí con la MISMA función que en entrenamiento, que es lo que
    garantiza consistencia entrenamiento/inferencia.
    """
    X = features_regresion(df_intensidades)[bundle["features"]]  # features en el orden congelado

    salida = pd.DataFrame(index=df_intensidades.index)    # DataFrame de predicciones
    for ley, modelo in bundle["modelos"].items():         # por cada ley del bundle
        salida[ley] = modelo.predict(X.values)            # predicción de esa ley
    return salida                                         # leyes estimadas


# ============================================================
# 4. BACKTEST WALK-FORWARD (evaluación honesta de los 3 cambios)
# ============================================================

def backtest(D: pd.DataFrame, frac_test=0.30, dias_ventana=DIAS_VENTANA_MOVIL,
             paso_reentreno_dias=PASO_REENTRENO_DIAS, usar_sesgo=True,
             leyes=LEYES_RECALIBRAR, devolver_corrector=False, corrector=None):
    """Simula la operación real: reentrenar cada N días y corregir sesgo en línea.

    Protocolo, para cada compósito del período de prueba:
      1. Si toca reentrenar (pasaron `paso_reentreno_dias`), se reajusta el modelo
         usando SOLO los últimos `dias_ventana` días anteriores a ese momento.
      2. Se predice el compósito actual (que el modelo nunca vio).
      3. Se aplica la corrección de sesgo estimada con los compósitos anteriores.
      4. Recién entonces se revela el valor real y se actualiza el corrector.

    El paso 4 después del 3 es lo que impide la fuga de información: el valor real
    del punto evaluado nunca participa de su propia corrección.

    Retorna
    -------
    DataFrame con ts, turno, y por cada ley: predicción cruda, corregida y real.
    Si devolver_corrector=True, devuelve (DataFrame, CorrectorSesgo). El corrector
    queda cargado con los residuos FUERA DE MUESTRA de los compósitos más
    recientes, que es justo lo que se quiere llevar a producción: un offset
    estimado sobre predicciones que el modelo no había visto.
    """
    D = D.sort_values("ts").reset_index(drop=True)        # orden temporal estricto
    corte = D["ts"].quantile(1 - frac_test)               # frontera entrenamiento / prueba
    idx_test = D.index[D["ts"] > corte]                   # índices del período de prueba

    # corrector de sesgo (CAMBIO 3); se puede inyectar uno distinto (p.ej.
    # CorrectorKalman) para compararlo en igualdad de condiciones -- por
    # defecto es exactamente el mismo objeto que antes, comportamiento
    # idéntico si no se pasa nada.
    if corrector is None:
        corrector = CorrectorSesgo()
    bundle = None                                         # bundle vigente (se reentrena periódicamente)
    ultimo_reentreno = None                               # fecha del último reentreno

    filas = []                                            # resultados del backtest
    for i in idx_test:                                    # recorre el período de prueba en orden
        fila = D.loc[i]                                   # compósito actual
        t = fila["ts"]                                    # su timestamp

        # --- 1. Reentrenar si toca (CAMBIO 2: ventana móvil) ---
        if (ultimo_reentreno is None or                   # primera vez
                (t - ultimo_reentreno).days >= paso_reentreno_dias):  # o ya venció el período
            ventana = ventana_de_entrenamiento(D, t, dias_ventana)     # solo datos ANTERIORES a t
            if len(ventana) >= 40:                        # exige un mínimo para que el ajuste tenga sentido
                bundle = entrenar(ventana, leyes)         # reajusta con la ventana móvil
                ultimo_reentreno = t                      # registra el reentreno

        if bundle is None:                                # todavía no hay modelo utilizable
            continue                                      # se salta este punto

        # --- 2. Predecir el punto actual (nunca visto por el modelo) ---
        pred = predecir(bundle, D.loc[[i]])               # predicción cruda

        registro = {"ts": t, "turno": fila["turno"]}      # fila de resultados
        for ley in bundle["modelos"]:                     # por cada ley estimada
            col_lab = MAPA_LEY_COMPOSITO[ley]             # columna real de laboratorio
            cruda = float(pred[ley].iloc[0])              # predicción sin corregir

            # --- 3. Corregir sesgo con la historia PREVIA (CAMBIO 3) ---
            corregida = (corrector.corregir(ley, cruda, fila["turno"])
                         if usar_sesgo else cruda)        # aplica offset aprendido

            registro[f"{ley}_cruda"] = cruda              # guarda la cruda (para comparar)
            registro[f"{ley}_corr"] = corregida           # guarda la corregida
            registro[f"{ley}_real"] = float(fila[col_lab])  # valor real del compósito

            # --- 4. Recién ahora se revela el real y se actualiza el corrector ---
            corrector.actualizar(ley, cruda, fila[col_lab], fila["turno"], ts=t)  # sin fuga: es posterior al paso 3

        filas.append(registro)                            # acumula el resultado

    R = pd.DataFrame(filas)                               # tabla del backtest
    return (R, corrector) if devolver_corrector else R    # opcionalmente devuelve el corrector ya cargado


def metricas(R: pd.DataFrame, leyes=LEYES_RECALIBRAR) -> pd.DataFrame:
    """Calcula R2, MAE, correlación, pendiente y sesgo del backtest.

    Reporta la versión cruda y la corregida por sesgo, para que se vea cuánto
    aporta el CAMBIO 3 por separado.
    """
    filas = []                                            # una fila por (ley, variante)
    for ley in leyes:                                     # por cada ley evaluada
        if f"{ley}_real" not in R.columns:                # ley no presente en el backtest
            continue                                      # se salta
        y = R[f"{ley}_real"].values                       # valores reales del compósito

        for variante, col in [("cruda", f"{ley}_cruda"), ("corregida", f"{ley}_corr")]:
            p = R[col].values                             # predicciones de esa variante
            filas.append({
                "ley": ley,                               # nombre de la ley
                "variante": variante,                     # cruda o corregida por sesgo
                "n": len(y),                              # nº de compósitos evaluados
                "R2": r2_score(y, p),                     # coeficiente de determinación
                "MAE": mean_absolute_error(y, p),         # error absoluto medio
                "corr": float(np.corrcoef(y, p)[0, 1]),   # correlación (métrica del techo de colocación triple)
                "pendiente": float(np.polyfit(y, p, 1)[0]),  # pendiente de predicho vs real
                "sesgo": float(p.mean() - y.mean()),      # sesgo medio
            })
    return pd.DataFrame(filas)                            # tabla de métricas


# ============================================================
# CLI — ejecución independiente de la etapa
# ============================================================
def _main():
    import argparse
    import joblib

    from .config import DATA_PROCESSED, MODELS_DIR, ART_CALIB_COMPOSITO

    ap = argparse.ArgumentParser(description="Recalibración contra el compósito de 12 h (cambios 1-3)")
    ap.add_argument("--intensidades", default=str(DATA_PROCESSED / "intensidades_cobre.csv"),
                    help="CSV de intensidades del analizador (stream completo)")
    ap.add_argument("--composito", default=str(LAB_COMPOSITO),
                    help="CSV del compósito de 12 h exportado de PI (tags 7100AIP10*MAN)")
    ap.add_argument("--backtest", action="store_true",
                    help="Corre el backtest walk-forward y reporta métricas (no guarda modelos)")
    ap.add_argument("--entrenar-final", action="store_true",
                    help="Entrena con la ventana móvil más reciente y guarda el bundle")
    ap.add_argument("--dias-ventana", type=int, default=DIAS_VENTANA_MOVIL,
                    help="Ancho de la ventana móvil de entrenamiento, en días")
    ap.add_argument("--output-bundle", default=str(MODELS_DIR / ART_CALIB_COMPOSITO))
    args = ap.parse_args()

    inten = cargar_intensidades(args.intensidades)        # stream de intensidades
    lab = cargar_composito(args.composito)                # compósito de 12 h
    D = alinear_con_composito(inten, lab)                 # dataset supervisado alineado
    print(f"Compósitos alineados: {len(D)}  ({D.ts.min()} -> {D.ts.max()})")

    if args.backtest:                                     # --- modo evaluación ---
        R = backtest(D, dias_ventana=args.dias_ventana)   # backtest walk-forward
        print(f"\nPuntos evaluados fuera de muestra: {len(R)}")
        print(metricas(R).round(3).to_string(index=False))

    if args.entrenar_final:                               # --- modo producción ---
        fin = D["ts"].max() + pd.Timedelta(seconds=1)     # referencia = último dato disponible
        ventana = ventana_de_entrenamiento(D, fin, args.dias_ventana)  # ventana móvil más reciente
        bundle = entrenar(ventana)                        # entrena el bundle final
        joblib.dump(bundle, args.output_bundle)           # serializa
        print(f"\nBundle entrenado con {bundle['n_entrenamiento']} compósitos "
              f"({bundle['ts_min']} -> {bundle['ts_max']})")
        print(f"Guardado en: {args.output_bundle}")


if __name__ == "__main__":
    _main()
