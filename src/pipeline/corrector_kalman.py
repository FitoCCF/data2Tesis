#!/usr/bin/env python3
# ============================================================
# src/pipeline/corrector_kalman.py — Corrector de sesgo por filtro de Kalman
# ============================================================
# Alternativa a CorrectorSesgo (media móvil de N=20, calibracion_composito.py):
# viola la regla del proyecto "suavizado de datos operacionales: EWMA, nunca
# media móvil simple". El sesgo real deambula ±1 punto porcentual entre
# trimestres (ver docstring de CorrectorSesgo) -- un estado que cambia lento
# en el tiempo es exactamente el caso de uso de un filtro de Kalman.
#
# Modelo de estado (por ley, y por turno si por_turno=True):
#   sesgo_t = sesgo_{t-1} + w_t            w_t ~ N(0, Q * dt_dias)   (caminata aleatoria)
#   residuo_t = sesgo_t + v_t              v_t ~ N(0, R)             (observación ruidosa)
#
# Q se escala por el tiempo transcurrido desde la última observación (dt_dias):
# si pasó más tiempo, más incertidumbre se acumula sobre el estado -- esto es
# lo que una ventana fija de N muestras NO puede hacer (trata muestreo
# irregular como si fuera regular).
#
# Misma interfaz pública que CorrectorSesgo (actualizar/sesgo/corregir), para
# poder inyectarlo en calibracion_composito.backtest(corrector=...) sin tocar
# esa función.
# ============================================================

from .config import N_MUESTRAS_SESGO, SESGO_POR_TURNO


class CorrectorKalman:
    """Corrector de sesgo con un filtro de Kalman escalar por (ley, turno).

    Parámetros
    ----------
    q_por_dia : varianza de proceso por día -- cuánto se permite mover el
        sesgo "verdadero" entre una observación y la siguiente. Chico = más
        suavizado (memoria larga); grande = más reactivo (memoria corta).
        Es el hiperparámetro a barrer en el backtest, igual que se barrió
        dias_ventana/paso_reentreno_dias para la recalibración de Fe.
    r : varianza de observación (ruido de cada residuo individual). Se deja
        fija en 1.0 -- en un Kalman escalar solo importa la RAZÓN q/r, así que
        fijar r y barrer q_por_dia ya cubre el espacio de comportamientos.
    var_inicial : varianza del prior antes de la primera observación (alta =
        "no sé nada todavía", se corrige rápido con la primera observación).
    """

    def __init__(self, q_por_dia: float = 0.02, r: float = 1.0,
                var_inicial: float = 10.0,
                n_muestras=N_MUESTRAS_SESGO, por_turno=SESGO_POR_TURNO):
        self.q_por_dia = q_por_dia
        self.r = r
        self.var_inicial = var_inicial
        self.por_turno = por_turno
        self.n_muestras = n_muestras          # umbral mínimo de observaciones para confiar en el turno
        self.estado = {}                      # (ley, turno) -> [media, var, ultimo_ts, n_obs]
        self.estado_global = {}               # ley -> [media, var, ultimo_ts, n_obs]

    def _paso(self, almacen: dict, clave, residuo: float, ts):
        """Un paso predicción+actualización de Kalman sobre una clave dada."""
        media, var, ultimo_ts, n_obs = almacen.get(clave, (0.0, self.var_inicial, None, 0))

        # --- predicción: el estado no cambia de media, pero crece su incertidumbre ---
        dt_dias = 0.0
        if ultimo_ts is not None and ts is not None:
            dt_dias = max((ts - ultimo_ts).total_seconds() / 86400.0, 0.0)
        var_pred = var + self.q_por_dia * dt_dias

        # --- actualización: se combina la predicción con la observación nueva ---
        ganancia = var_pred / (var_pred + self.r)
        media = media + ganancia * (residuo - media)
        var = (1 - ganancia) * var_pred

        almacen[clave] = (media, var, ts, n_obs + 1)

    def actualizar(self, ley: str, prediccion: float, real: float, turno: str = "dia", ts=None):
        """Registra un residuo nuevo (mismo contrato que CorrectorSesgo.actualizar)."""
        residuo = float(prediccion) - float(real)
        if self.por_turno:
            self._paso(self.estado, (ley, turno), residuo, ts)
        self._paso(self.estado_global, ley, residuo, ts)

    def sesgo(self, ley: str, turno: str = "dia") -> float:
        """Sesgo estimado (media posterior del estado). 0.0 si aún no hay datos."""
        if self.por_turno:
            clave = (ley, turno)
            if clave in self.estado and self.estado[clave][3] >= 3:   # mínimo de 3, igual que CorrectorSesgo
                return float(self.estado[clave][0])
        if ley in self.estado_global:
            return float(self.estado_global[ley][0])
        return 0.0

    def corregir(self, ley: str, prediccion: float, turno: str = "dia") -> float:
        """Aplica la corrección: predicción - sesgo estimado."""
        return float(prediccion) - self.sesgo(ley, turno)
