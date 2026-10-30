#!/usr/bin/env python3
# ============================================================
# scripts/en_linea_excel.py — monitor "como el Excel de planta" (Courier_AUTO-C2.xlsm)
# ============================================================
# Compara, guardia por guardia, cuatro valores contra la misma referencia:
#   Outocal   promedio de guardia que calcula el DCS (tag _CAL), leído al
#             cierre de la guardia como lo hace la hoja Courier del Excel
#             (PIArcVal a las 18:30 para el turno A, 06:30 del día siguiente
#             para el B);
#   Modelo    estimación por lectura del modelo, promediada en la MISMA guardia;
#   Kalman    el modelo llevado a escala balde con el Kalman (q elegido en
#             train, sesgo causal: solo compósitos ya publicados);
#   Lab       compósito de laboratorio de la guardia.
# Guardia A = [06:30, 18:30) ↔ compósito de las 07:30; guardia B =
# [18:30, 06:30+1d) ↔ compósito de las 19:30. Es decir, [t−1 h, t+11 h).
#
# Reglas tomadas del Excel (hojas TABLA, Calculos y Portada):
#   DIF = Lab − X; %ERROR = |DIF| / Lab · 100;
#   el DIF se ignora si el valor del courier está bajo el mínimo de planta
#   (concentrado Cu 18 / Fe 22 / Mo 0.3; cabeza Cu 0.2 / Fe 1 / Mo 0.004);
#   desviación estándar de DIF y de %ERROR sobre los últimos 31 días;
#   error aceptable = promedio del lab × 6 % (concentrado) u 11 % (cabeza).
#   El Excel no define tolerancia de Fe: aquí se usa el mismo % del stream.
#
# NO toca los archivos de los otros monitores: solo LEE la estimación por
# lectura (courier_estimado.csv / courier_cabeza_L2_pi.csv) y el compósito
# que ellos mantienen al día. Del PI trae únicamente los tres tags _CAL.
#
# Salidas: data/final/excel_<perfil>.csv, reports/excel_<perfil>.html
#
#   PI_GATEWAY_HOST=<ip> PI_GATEWAY_PORT=<puerto> pixi run python -u scripts/en_linea_excel.py --perfil concentrado --cada 20
#   PI_GATEWAY_HOST=<ip> PI_GATEWAY_PORT=<puerto> pixi run python -u scripts/en_linea_excel.py --perfil cabeza --cada 20
#   pixi run python scripts/en_linea_excel.py --perfil concentrado --sin-extraer      # sin PI
#
# El perfil "cabeza" necesita los scripts de cabeza de la rama AlimConc.
# La IP de la pasarela va solo por entorno, nunca en un archivo.
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import html
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.config import (DATA_RAW, DATA_FINAL, REPORTS_DIR, ESTIMADO, LIMPIO_COMPOSITO,
                                 BLOQUE_RESAMPLE, MIN_BLOQUES_VENTANA)
from src.evaluacion.evaluar import mascara_lecturas, kalman_serie, metricas, periodo_de, Q_GRID
from src.evaluacion.descripcion import modelo_concentrado, modelo_cabeza
from src.evaluacion.figuras import SUPERFICIE, TINTA, TINTA_2, REJILLA, COLOR_LAB, COLOR_MODELO, COLOR_KALMAN

COLOR_OUTOCAL = "#c2338a"            # validado contra azul/naranja (dataviz validate_palette, all-pairs)
H = pd.Timedelta(hours=1)
INICIO, FIN = -1 * H, 11 * H         # guardia del compósito t: [t−1 h, t+11 h)
CIERRE = 11 * H                      # el Excel lee el _CAL al cierre de la guardia
TOL_CIERRE = H                       # el DCS lo escribe ~18:28 / 06:28 (a veces 18:30:00.5)
DIAS = 31                            # "ÚLTIMOS 31 DÍAS" de la Portada
LEYES = ["cu", "fe", "mo"]
SERIES = [("outo", "Outocal (DCS)"), ("est", "Modelo"), ("kal", "Modelo + Kalman")]


@dataclass
class Perfil:
    nombre: str
    titulo: str
    tags: dict                       # tag _CAL -> ley
    tol: float                       # error aceptable = media lab × tol
    minimo: dict                     # bajo este valor el Excel ignora el DIF
    dec: dict

    @property
    def raw(self) -> Path:
        return DATA_RAW / f"outocal_{self.nombre}_pi.csv"

    @property
    def salida(self) -> Path:
        return DATA_FINAL / f"excel_{self.nombre}.csv"


PERFILES = {
    "concentrado": Perfil("concentrado", "Concentrado colectivo",
                          {"_7100_AIP_CONC_CU_CAL": "cu", "_7100_AIP_CONC_FE_CAL": "fe", "_7100_AIP_CONC_MO_CAL": "mo"},
                          0.06, {"cu": 18, "fe": 22, "mo": 0.3}, {"cu": 2, "fe": 2, "mo": 3}),
    "cabeza": Perfil("cabeza", "Cabeza (alimentación rougher)",
                     {"_7100_AIP_HPRO_CU_CAL": "cu", "AIP_7100_HPRO_FE_CAL": "fe", "_7100_AIP_HPRO_MO_CAL": "mo"},
                     0.11, {"cu": 0.2, "fe": 1, "mo": 0.004}, {"cu": 3, "fe": 2, "mo": 4}),
}


def _leer(ruta, intentos: int = 4, **kw) -> pd.DataFrame:
    """Lee un CSV que otro monitor puede estar reescribiendo: si sale vacío o
    roto, espera y reintenta."""
    for i in range(intentos):
        try:
            d = pd.read_csv(ruta, **kw)
            if len(d):
                return d
        except Exception:
            if i == intentos - 1:
                raise
        time.sleep(15)
    raise RuntimeError(f"{ruta.name} vacío tras {intentos} intentos")


# --- estimación por lectura de cada perfil ---------------------------------
class FuenteConcentrado:
    """Lee la estimación por lectura que deja en_linea.py (etapa 6)."""

    minimo_ventana = MIN_BLOQUES_VENTANA                 # en bloques de 30 min

    def bloques(self) -> tuple[pd.DataFrame, pd.Timestamp]:
        est = _leer(ESTIMADO, parse_dates=["ts"])
        cols = {"pCu_est": "cu", "pFe_est": "fe", "pMo_est": "mo"}
        b = (est.loc[mascara_lecturas(est)].set_index("ts")[list(cols)].rename(columns=cols)
             .resample(BLOQUE_RESAMPLE).mean().dropna())
        return b, est.loc[est["fuente"] != "bd", "ts"].max()

    def lab(self) -> pd.DataFrame:
        return _leer(LIMPIO_COMPOSITO, parse_dates=["ts"]).set_index("ts")[LEYES]

    def periodo(self, ts: pd.Series) -> pd.Series:
        return periodo_de(ts)


class FuenteCabeza:
    """Mismo Ridge global que en_linea_cabeza.py (se ajusta una vez, solo train)."""

    def __init__(self):
        from scripts.en_linea_cabeza import MonitorCabeza, LINEA, DESDE_SALIDA
        from scripts.cabeza_intensidades import FEATS, MIN_LECTURAS
        self.modelos = MonitorCabeza(extraer=False).modelos
        self.linea, self.desde_test, self.feats = LINEA, DESDE_SALIDA, FEATS
        self.minimo_ventana = MIN_LECTURAS

    def bloques(self) -> tuple[pd.DataFrame, pd.Timestamp]:
        from scripts.cabeza_intensidades import lecturas
        for i in range(4):                           # el otro monitor puede estar escribiendo el crudo
            try:
                lect = lecturas(self.linea)
                break
            except Exception:
                if i == 3:
                    raise
                time.sleep(15)
        b = pd.DataFrame({l: self.modelos[l].predict(lect[self.feats]) for l in LEYES}, index=lect.index)
        return b, lect.index.max()                   # por lectura: la ventana pide MIN_LECTURAS lecturas

    def lab(self) -> pd.DataFrame:
        from src.pipeline.config import RAW_CABEZA
        return _leer(RAW_CABEZA, parse_dates=["ts"]).set_index("ts")[LEYES]

    def periodo(self, ts: pd.Series) -> pd.Series:
        return pd.Series(np.where(ts < self.desde_test, "train", "test"), index=ts.index)


# --- monitor -----------------------------------------------------------------
class MonitorExcel:
    def __init__(self, perfil: Perfil, extraer: bool = True):
        self.p, self.extraer = perfil, extraer
        self.fuente = FuenteCabeza() if perfil.nombre == "cabeza" else FuenteConcentrado()
        self.modelo = modelo_cabeza("L2") if perfil.nombre == "cabeza" else modelo_concentrado()
        print(self.modelo)

    def adquirir(self) -> None:
        """Trae los _CAL nuevos (incremental). Si la pasarela no responde, sigue
        con lo que ya hay: el resto del ciclo no depende del PI."""
        from src.acquisition.from_pi import cliente, _a_ancho
        from scripts.en_linea import MonitorEnLinea
        ruta = self.p.raw
        desde = (pd.Timestamp(pd.read_csv(ruta, usecols=["ts"])["ts"].max()) - pd.Timedelta(days=2)
                 if ruta.exists() else pd.Timestamp("2025-12-15"))
        try:
            nuevo = _a_ancho(cliente().recorded(list(self.p.tags), desde.strftime("%Y-%m-%d %H:%M:%S"),
                                                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                chunk_dias=90), self.p.tags)
        except Exception as e:
            print(f"  (Outocal no actualizado: {type(e).__name__}: {str(e)[:150]})")
            return
        if not ruta.exists():
            nuevo.to_csv(ruta, index=False)
            print(f"  {ruta.name}: creado con {len(nuevo)} filas")
            return
        n, hasta = MonitorEnLinea._unir(None, ruta, nuevo)
        print(f"  {ruta.name}: +{n} filas, hasta {hasta}")

    def _outocal(self, marcas: pd.DatetimeIndex) -> pd.DataFrame:
        """_CAL escrito más cerca del cierre de cada guardia (±1 h)."""
        o = pd.read_csv(self.p.raw, parse_dates=["ts"]).set_index("ts").sort_index()
        cierre = pd.DataFrame({"cierre": (marcas + CIERRE).values})       # marcas ya vienen ordenadas
        out = pd.DataFrame(index=marcas)
        for l in LEYES:
            s = o[l].dropna().rename("v").rename_axis("t_cal").reset_index()
            m = pd.merge_asof(cierre, s, left_on="cierre", right_on="t_cal",
                              direction="nearest", tolerance=TOL_CIERRE)
            out[l] = m["v"].values
        return out

    def estimar(self) -> pd.DataFrame:
        b, ultima = self.fuente.bloques()
        lab = self.fuente.lab()
        lab = lab[lab.index >= b.index.min() - INICIO]
        marcas, m, ahora = list(lab.index), lab.index.max() + 12 * H, pd.Timestamp.now()
        while m + INICIO <= ahora:                       # pendientes hasta la guardia en curso
            marcas.append(m)
            m += 12 * H
        filas = []
        for t in marcas:
            w = b.loc[t + INICIO: t + FIN - pd.Timedelta(seconds=1)]
            fila = {"ts": t, "n": len(w)}
            fila.update({f"{l}_est": (w[l].mean() if len(w) >= self.fuente.minimo_ventana else np.nan) for l in LEYES})
            fila.update({l: lab.at[t, l] if t in lab.index else np.nan for l in LEYES})
            filas.append(fila)
        A = pd.DataFrame(filas).sort_values("ts").reset_index(drop=True)
        A["fecha"] = A["ts"].dt.normalize()
        A["turno"] = np.where(A["ts"].dt.hour < 12, "dia", "noche")   # nombre que espera el Kalman
        A["periodo"] = self.fuente.periodo(A["ts"]).values
        for l in LEYES:                                  # q elegido SOLO con train; sin estimación no se actualiza
            B = A[A[f"{l}_est"].notna()]
            tr = (B["periodo"] == "train") & B[l].notna()
            r2q = {q: metricas(B.loc[tr, l], kalman_serie(B, l, q)[0][tr]).get("R2", -np.inf) for q in Q_GRID}
            A[f"{l}_kal"] = kalman_serie(B, l, max(r2q, key=r2q.get))[0].reindex(A.index)
        o = self._outocal(pd.DatetimeIndex(A["ts"]))
        for l in LEYES:
            A[f"{l}_outo"] = o[l].values
        A["publicado"] = A["cu"].notna()
        A["guardia_cerrada"] = A["ts"] + FIN <= ahora
        A["ultima_lectura"] = ultima
        A = A[A["periodo"] != "train"].reset_index(drop=True)
        A.to_csv(self.p.salida, index=False)
        return A

    def ciclo(self) -> None:
        t0 = time.time()
        print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] ciclo Excel ({self.p.nombre})")
        if self.extraer:
            self.adquirir()
        A = self.estimar()
        print(f"  listo en {time.time() - t0:.0f} s; última lectura del courier: {A['ultima_lectura'].iloc[0]}")
        mostrar(A, self.p)
        try:
            tablero(A, self.p, self.modelo)
            print(f"  tablero: reports/excel_{self.p.nombre}.html")
        except Exception as e:
            print(f"  (tablero no generado: {type(e).__name__}: {str(e)[:200]})")


# --- reglas del Excel ----------------------------------------------------------
def dif(A: pd.DataFrame, l: str, s: str, p: Perfil) -> pd.Series:
    """DIF = Lab − X, en blanco si X está bajo el mínimo de planta."""
    x = A[f"{l}_{s}"]
    return (A[l] - x).where(x >= p.minimo[l])


def resumen(A: pd.DataFrame, p: Perfil) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Bloque "ÚLTIMOS 31 DÍAS": sobre las guardias publicadas de los últimos 31 días."""
    fin = A.loc[A["publicado"], "ts"].max()
    V = A[A["publicado"] & (A["ts"] > fin - pd.Timedelta(days=DIAS))]
    filas = []
    for l in LEYES:
        tol = V[l].mean() * p.tol
        comun = pd.concat([dif(V, l, s, p) for s, _ in SERIES], axis=1).dropna().index   # mismas guardias
        for s, nom in SERIES:
            d = dif(V, l, s, p).loc[comun]
            e = d.abs() / V.loc[d.index, l] * 100
            filas.append({"ley": l, "serie": s, "nombre": nom, "n": len(d), "tol": tol,
                          "sd_dif": d.std(), "sd_err": e.std(), "mae": d.abs().mean(), "sesgo": d.mean(),
                          "acept": (d.abs() < tol).mean() * 100 if len(d) else np.nan})
    return pd.DataFrame(filas), fin


def mostrar(A: pd.DataFrame, p: Perfil, n: int = 6) -> None:
    filas = []
    for f in A.tail(n).itertuples(index=False):
        fila = {"guardia": f"{f.ts:%m-%d} {'A' if f.ts.hour < 12 else 'B'}",
                "estado": "lab" if f.publicado else ("cerrada" if f.guardia_cerrada else "en curso")}
        for l in LEYES:
            d = p.dec[l]
            o, k, lab = getattr(f, f"{l}_outo"), getattr(f, f"{l}_kal"), getattr(f, l)
            txt = f"O {o:.{d}f} · K {k:.{d}f}" if not (np.isnan(o) and np.isnan(k)) else "sin datos"
            fila[l.capitalize()] = txt.replace("nan", "—") + ("" if np.isnan(lab) else f" | lab {lab:.{d}f}")
        filas.append(fila)
    print(pd.DataFrame(filas).to_string(index=False))


# --- tablero -------------------------------------------------------------------
def tendencia(A: pd.DataFrame, p: Perfil, desde: pd.Timestamp) -> list:
    """Puntos por guardia para los gráficos de la página (se dibujan en el navegador
    para poder prender y apagar cada serie). Bajo el mínimo de planta el valor no
    se dibuja, igual que el Excel no lo compara."""
    pts = []
    for f in A[A["ts"] >= desde].itertuples(index=False):
        pt = {"t": f"{f.ts:%Y-%m-%dT%H:%M}", "g": f"{f.ts:%d-%m} {'A' if f.ts.hour < 12 else 'B'}"}
        for l in LEYES:
            for suf in ("", "_outo", "_est", "_kal"):
                v = getattr(f, f"{l}{suf}")
                pt[f"{l}{suf}"] = None if np.isnan(v) or (suf and v < p.minimo[l]) else round(float(v), 5)
        pts.append(pt)
    return pts


def tablero(A: pd.DataFrame, p: Perfil, modelo: str = "") -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    R, fin = resumen(A, p)
    desde = (fin if pd.notna(fin) else A["ts"].max()) - pd.Timedelta(days=DIAS)
    puntos = tendencia(A, p, desde)

    # datos de la Portada por fecha (la página elige la fecha en el navegador)
    portada = {}
    for f in A.itertuples(index=False):
        dia = portada.setdefault(f"{f.fecha:%Y-%m-%d}", {})
        dia["A" if f.ts.hour < 12 else "B"] = {
            "estado": "lab" if f.publicado else ("guardia cerrada" if f.guardia_cerrada else "en curso"),
            **{f"{l}_{s}": (None if np.isnan(v := getattr(f, f"{l}_{s}")) else round(float(v), 5))
               for l in LEYES for s in ("outo", "est", "kal")},
            **{l: (None if np.isnan(v := getattr(f, l)) else float(v)) for l in LEYES}}
    con_lab = A.loc[A["publicado"], "fecha"]
    defecto = f"{(con_lab.max() if len(con_lab) else A['fecha'].max()):%Y-%m-%d}"

    # bloque "últimos 31 días"
    filas31 = []
    for l in LEYES:
        r = R[R["ley"] == l]
        mejor = r["mae"].idxmin() if r["mae"].notna().any() else None
        d = p.dec[l]
        for i, x in r.iterrows():
            estrella = " ★" if i == mejor else ""
            filas31.append(
                f"<tr><td>{l.capitalize()}</td><td class='i'>{x['nombre']}{estrella}</td><td>{x['n']}</td>"
                f"<td>{x['sd_dif']:.{d}f}</td><td>{x['sd_err']:.1f}</td><td>{x['mae']:.{d}f}</td>"
                f"<td>{x['sesgo']:+.{d}f}</td><td>{x['acept']:.0f} %</td><td class='d'>±{x['tol']:.{d}f}</td></tr>")

    # tabla de guardias (como la hoja TABLA), más reciente arriba
    filas_t = []
    for f in A[A["ts"] >= desde].iloc[::-1].itertuples(index=False):
        celdas = [f"{f.ts:%d-%m}", "A" if f.ts.hour < 12 else "B"]
        for l in LEYES:
            d = p.dec[l]
            lab = getattr(f, l)
            celdas.append("—" if np.isnan(lab) else f"<b>{lab:.{d}f}</b>")
            for s, _ in SERIES:
                v = getattr(f, f"{l}_{s}")
                if np.isnan(v):
                    celdas.append("—")
                elif np.isnan(lab) or v < p.minimo[l]:
                    celdas.append(f"{v:.{d}f}")
                else:
                    celdas.append(f"{v:.{d}f} <span class='d'>({lab - v:+.{d}f})</span>")
        filas_t.append("<tr>" + "".join(f"<td>{c}</td>" for c in celdas) + "</tr>")

    ultima = A["ultima_lectura"].iloc[-1]
    horas = (pd.Timestamp.now() - pd.Timestamp(ultima)) / H
    aviso = f"<p class='aviso'>⚠ El courier no entrega lecturas hace {horas:.1f} h</p>" if horas > 1 else ""
    sub = "".join(f"<th>{n}</th>" for _, n in [("", "Lab")] + SERIES)
    tol_js = {l: p.tol for l in LEYES}
    (REPORTS_DIR / f"excel_{p.nombre}.html").write_text(f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta http-equiv="refresh" content="120">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{p.titulo} vs Excel</title>
<style>
  :root {{ --sup:{SUPERFICIE}; --tinta:{TINTA}; --t2:{TINTA_2}; --rej:{REJILLA}; }}
  body {{ background:var(--sup); color:var(--tinta); font:14px system-ui, sans-serif; margin:16px; }}
  h1 {{ font-size:19px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:22px 0 6px; }}
  img {{ max-width:100%; height:auto; border:1px solid var(--rej); }}
  table {{ border-collapse:collapse; font-variant-numeric:tabular-nums; }}
  th, td {{ padding:4px 9px; border-bottom:1px solid var(--rej); text-align:right; white-space:nowrap; }}
  th {{ color:var(--t2); font-weight:600; }} th.g {{ text-align:center; border-bottom:2px solid var(--rej); }}
  td.i {{ text-align:left; }} .d {{ color:var(--t2); }} .aviso {{ font-weight:600; }}
  .pie {{ color:var(--t2); font-size:12px; max-width:900px; }}
  .tarjetas {{ display:flex; flex-wrap:wrap; gap:16px; }}
  .tarjeta {{ border:1px solid var(--rej); border-radius:6px; padding:8px 10px; overflow-x:auto; max-width:100%; }}
  .tarjeta h3 {{ margin:0 0 4px; font-size:15px; }}
  .fuera {{ font-weight:700; }}
  .modelo {{ font-weight:600; margin:0 0 4px; }}
  .scroll {{ overflow-x:auto; max-width:100%; }}
  select {{ font:inherit; padding:2px 6px; }}
  .checks {{ display:flex; flex-wrap:wrap; gap:6px 18px; margin:4px 0 8px; }}
  .checks label {{ display:flex; align-items:center; gap:6px; cursor:pointer; user-select:none; }}
  .muestra {{ display:inline-block; width:22px; height:3px; border-radius:2px; }}
  .graf {{ border:1px solid var(--rej); border-radius:6px; margin-bottom:10px; background:var(--sup); }}
  .graf svg {{ display:block; width:100%; height:auto; }}
  .graf text {{ font:11px system-ui, sans-serif; fill:var(--t2); }}
  .tip {{ position:fixed; pointer-events:none; background:var(--sup); border:1px solid var(--rej); border-radius:6px;
         padding:6px 8px; font-size:12px; box-shadow:0 2px 6px rgba(0,0,0,.12); font-variant-numeric:tabular-nums; }}
  .tip td {{ padding:1px 6px; border:none; }}
</style></head><body>
<h1>{p.titulo} — comparación Courier vs Laboratorio, como el Excel de planta</h1>
<p class="modelo">{html.escape(modelo)}</p>
<p class="d">Última lectura del courier {html.escape(f'{pd.Timestamp(ultima):%d-%m %H:%M}')} ·
actualizado {datetime.now():%d-%m %H:%M} · DIF = Lab − valor · guardia A 06:30–18:30, B 18:30–06:30</p>
{aviso}

<h2>Portada — fecha <select id="fecha"></select></h2>
<div class="tarjetas" id="portada"></div>
<p class="pie">✖ = |DIF| mayor que el error aceptable (±{p.tol:.0%} del promedio del laboratorio de los últimos
{DIAS} días, criterio de la hoja Calculos). "—" = sin dato o bajo el mínimo de planta.</p>

<h2>Últimos {DIAS} días (hasta el compósito del {fin:%d-%m %H:%M})</h2>
<div class="scroll"><table><thead><tr><th>Ley</th><th class="i">Valor</th><th>n</th><th>Desv. est. DIF</th>
<th>Desv. est. %error</th><th>Error medio |DIF|</th><th>Sesgo (Lab − X)</th><th>Aceptable</th><th>Tolerancia</th></tr></thead>
<tbody>{''.join(filas31)}</tbody></table></div>
<p class="pie">Las tres filas de cada ley usan las mismas guardias (las que tienen los tres valores y el lab), para
que la comparación sea pareja. ★ = menor error medio en esa ley. "Aceptable" = % de guardias con |DIF| bajo la tolerancia
(el gráfico de torta de Graf3 del Excel). El Excel solo define tolerancia para Cu y Mo; para Fe se usa el mismo %.</p>

<h2>Tendencias (últimos {DIAS} días)</h2>
<div class="checks" id="checks"></div>
<div id="graficos"></div>
<div id="tip" class="tip" hidden></div>

<h2>Tabla por guardia</h2>
<div class="scroll"><table><thead>
<tr><th rowspan="2">fecha</th><th rowspan="2">turno</th>
<th colspan="4" class="g">Cu</th><th colspan="4" class="g">Fe</th><th colspan="4" class="g">Mo</th></tr>
<tr>{sub * 3}</tr></thead><tbody>{''.join(filas_t)}</tbody></table></div>
<p class="pie">Entre paréntesis, DIF = Lab − valor. Outocal = tag _CAL del DCS al cierre de la guardia. Modelo =
promedio de la estimación por lectura en la misma guardia. Kalman = modelo + sesgo aprendido de los compósitos
ya publicados. Los checks de las tendencias se recuerdan en este navegador. La página se recarga cada 2 min; los datos cambian cuando el monitor termina un ciclo.</p>

<script>
const D = {json.dumps(portada)};
const DEC = {json.dumps(p.dec)}, TOL = {json.dumps(tol_js)};
const PROM = {json.dumps({l: (None if np.isnan(v := A.loc[A['publicado'] & (A['ts'] > fin - pd.Timedelta(days=DIAS)), l].mean()) else float(v)) for l in LEYES})};
const MIN = {json.dumps(p.minimo)};
const SER = {json.dumps(SERIES)};
const sel = document.getElementById("fecha");
Object.keys(D).sort().reverse().forEach(k => sel.add(new Option(k.split("-").reverse().join("-"), k)));
let ini = "{defecto}";
try {{ const h = decodeURIComponent(location.hash.slice(1)); if (D[h]) ini = h; }} catch (e) {{}}
sel.value = ini;
sel.onchange = () => {{ location.hash = sel.value; pinta(); }};
const f = (v, d) => v === null || v === undefined ? "—" : v.toFixed(d);
function celda(g, l, s) {{
  if (!g) return "<td>—</td><td>—</td><td>—</td>";
  const v = g[l + "_" + s], lab = g[l], d = DEC[l];
  if (v === null || lab === null || v < MIN[l]) return `<td>${{f(v, d)}}</td><td>—</td><td>—</td>`;
  const dif = lab - v, pct = Math.abs(dif) / lab * 100, tol = PROM[l] === null ? null : PROM[l] * TOL[l];
  const fuera = tol !== null && Math.abs(dif) > tol;
  return `<td>${{f(v, d)}}</td><td class="${{fuera ? "fuera" : ""}}">${{(dif >= 0 ? "+" : "") + dif.toFixed(d)}}${{fuera ? " ✖" : ""}}</td><td>${{pct.toFixed(1)}}</td>`;
}}
function pinta() {{
  const dia = D[sel.value] || {{}}, A = dia.A, B = dia.B;
  let h = "";
  for (const l of ["cu", "fe", "mo"]) {{
    const lab = g => g ? `<td><b>${{f(g[l], DEC[l])}}</b></td><td></td><td></td>` : "<td>—</td><td></td><td></td>";
    let filas = `<tr><td class="i">Laboratorio</td>${{lab(A)}}${{lab(B)}}</tr>`;
    for (const [s, nom] of SER) filas += `<tr><td class="i">${{nom}}</td>${{celda(A, l, s)}}${{celda(B, l, s)}}</tr>`;
    const est = g => g ? g.estado : "sin guardia";
    h += `<div class="tarjeta"><h3>%${{l[0].toUpperCase() + l.slice(1)}}</h3><table><thead>
      <tr><th></th><th colspan="3" class="g">Turno A <span class="d">(${{est(A)}})</span></th><th colspan="3" class="g">Turno B <span class="d">(${{est(B)}})</span></th></tr>
      <tr><th></th><th>valor</th><th>DIF</th><th>%dif</th><th>valor</th><th>DIF</th><th>%dif</th></tr></thead>
      <tbody>${{filas}}</tbody></table></div>`;
  }}
  document.getElementById("portada").innerHTML = h;
}}
pinta();

// --- tendencias con series que se prenden y apagan ---
const P = {json.dumps(puntos)};
const SERIES_T = [
  {{k: "", nom: "Laboratorio (compósito)", color: "{COLOR_LAB}", ancho: 2.4}},
  {{k: "_outo", nom: "Outocal (DCS, _CAL)", color: "{COLOR_OUTOCAL}", ancho: 1.8}},
  {{k: "_est", nom: "Modelo", color: "{COLOR_MODELO}", ancho: 1.5}},
  {{k: "_kal", nom: "Modelo + Kalman", color: "{COLOR_KALMAN}", ancho: 2.0}},
  {{k: "tol", nom: "Error aceptable (±{p.tol:.0%} del lab)", color: "{TINTA_2}", ancho: 0}}];
let vis = {{"": true, "_outo": true, "_est": true, "_kal": true, "tol": true}};
try {{ Object.assign(vis, JSON.parse(localStorage.getItem("excel_series_{p.nombre}") || "{{}}")); }} catch (e) {{}}
const checks = document.getElementById("checks");
for (const s of SERIES_T) {{
  const lb = document.createElement("label");
  const sw = s.k === "tol" ? `<span class="muestra" style="height:10px;background:${{s.color}};opacity:.18"></span>`
                           : `<span class="muestra" style="background:${{s.color}}"></span>`;
  lb.innerHTML = `<input type="checkbox" ${{vis[s.k] ? "checked" : ""}}> ${{sw}} ${{s.nom}}`;
  lb.querySelector("input").onchange = e => {{
    vis[s.k] = e.target.checked;
    try {{ localStorage.setItem("excel_series_{p.nombre}", JSON.stringify(vis)); }} catch (e2) {{}}
    dibuja();
  }};
  checks.appendChild(lb);
}}
const tip = document.getElementById("tip");
function dibuja() {{
  const W = 1100, H = 230, m = {{l: 52, r: 14, t: 12, b: 26}};
  const ts = P.map(p => new Date(p.t).getTime());
  const x0 = Math.min(...ts), x1 = Math.max(...ts);
  const X = t => m.l + (t - x0) / Math.max(x1 - x0, 1) * (W - m.l - m.r);
  let html = "";
  for (const l of ["cu", "fe", "mo"]) {{
    const labs = P.map(p => p[l]).filter(v => v !== null);
    const tol = labs.length ? labs.reduce((a, b) => a + b, 0) / labs.length * TOL[l] : 0;
    let vals = [];
    for (const s of SERIES_T) {{
      if (!vis[s.k]) continue;
      if (s.k === "tol") labs.forEach(v => vals.push(v - tol, v + tol));
      else P.forEach(p => {{ if (p[l + s.k] !== null) vals.push(p[l + s.k]); }});
    }}
    if (!vals.length) vals = labs.length ? labs : [0, 1];
    let y0 = Math.min(...vals), y1 = Math.max(...vals);
    const pad = (y1 - y0) * 0.08 || Math.abs(y0) * 0.05 || 1; y0 -= pad; y1 += pad;
    const Y = v => m.t + (y1 - v) / (y1 - y0) * (H - m.t - m.b);
    let g = "";
    // rejilla y eje y
    for (let i = 0; i <= 4; i++) {{
      const v = y0 + (y1 - y0) * i / 4, y = Y(v);
      g += `<line x1="${{m.l}}" x2="${{W - m.r}}" y1="${{y}}" y2="${{y}}" stroke="{REJILLA}"/>`;
      g += `<text x="${{m.l - 6}}" y="${{y + 4}}" text-anchor="end">${{v.toFixed(DEC[l])}}</text>`;
    }}
    // eje x: un rótulo cada 3 días
    const dia = 86400000;
    for (let t = Math.ceil(x0 / dia) * dia; t <= x1; t += 3 * dia) {{
      const d = new Date(t + new Date(t).getTimezoneOffset() * 60000);
      g += `<text x="${{X(t)}}" y="${{H - 6}}" text-anchor="middle">${{String(d.getDate()).padStart(2, "0")}}-${{String(d.getMonth() + 1).padStart(2, "0")}}</text>`;
    }}
    // banda de tolerancia alrededor del lab
    if (vis.tol) {{
      let tramo = [];
      const cierra = () => {{
        if (tramo.length > 1) {{
          const sup = tramo.map(([t, v]) => `${{X(t)}},${{Y(v + tol)}}`).join(" ");
          const inf = tramo.slice().reverse().map(([t, v]) => `${{X(t)}},${{Y(v - tol)}}`).join(" ");
          g += `<polygon points="${{sup}} ${{inf}}" fill="{TINTA_2}" opacity="0.12"/>`;
        }}
        tramo = [];
      }};
      P.forEach((p, i) => p[l] === null ? cierra() : tramo.push([ts[i], p[l]]));
      cierra();
    }}
    // series: línea cortada donde falta el dato, y marcador en cada guardia
    for (const s of SERIES_T) {{
      if (s.k === "tol" || !vis[s.k]) continue;
      let d = "", pen = false;
      P.forEach((p, i) => {{
        const v = p[l + s.k];
        if (v === null) {{ pen = false; return; }}
        d += `${{pen ? "L" : "M"}}${{X(ts[i]).toFixed(1)}},${{Y(v).toFixed(1)}}`; pen = true;
      }});
      g += `<path d="${{d}}" fill="none" stroke="${{s.color}}" stroke-width="${{s.ancho}}" stroke-linejoin="round"/>`;
      P.forEach((p, i) => {{
        const v = p[l + s.k];
        if (v !== null) g += `<circle cx="${{X(ts[i])}}" cy="${{Y(v)}}" r="${{s.k === "" ? 3.2 : 2.6}}" fill="${{s.color}}" stroke="{SUPERFICIE}" stroke-width="1"/>`;
      }});
    }}
    g += `<line class="cruz" x1="0" x2="0" y1="${{m.t}}" y2="${{H - m.b}}" stroke="{TINTA_2}" stroke-dasharray="3 3" visibility="hidden"/>`;
    g += `<rect class="caza" x="${{m.l}}" y="${{m.t}}" width="${{W - m.l - m.r}}" height="${{H - m.t - m.b}}" fill="transparent"/>`;
    html += `<div class="graf" data-ley="${{l}}"><svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="Tendencia de %${{l}}">
      <text x="${{m.l}}" y="${{m.t + 2}}" dy="0" style="font-weight:600;fill:{TINTA}">%${{l[0].toUpperCase() + l.slice(1)}}</text>${{g}}</svg></div>`;
  }}
  const cont = document.getElementById("graficos");
  cont.innerHTML = html;
  // tooltip: guardia más cercana al cursor
  cont.querySelectorAll(".graf").forEach(div => {{
    const l = div.dataset.ley, svg = div.querySelector("svg"), cruz = svg.querySelector(".cruz");
    const caza = svg.querySelector(".caza");
    caza.addEventListener("mousemove", ev => {{
      const r = svg.getBoundingClientRect(), xs = (ev.clientX - r.left) / r.width * W;
      let i = 0, best = Infinity;
      ts.forEach((t, j) => {{ const dd = Math.abs(X(t) - xs); if (dd < best) {{ best = dd; i = j; }} }});
      cruz.setAttribute("x1", X(ts[i])); cruz.setAttribute("x2", X(ts[i])); cruz.setAttribute("visibility", "visible");
      const p = P[i];
      let filas = "";
      for (const s of SERIES_T) {{
        if (s.k === "tol" || !vis[s.k]) continue;
        const v = p[l + s.k];
        const dif = (s.k && v !== null && p[l] !== null) ? ` <span class="d">(${{(p[l] - v >= 0 ? "+" : "") + (p[l] - v).toFixed(DEC[l])}})</span>` : "";
        filas += `<tr><td><span class="muestra" style="background:${{s.color}}"></span> ${{s.nom}}</td><td style="text-align:right">${{f(v, DEC[l])}}${{dif}}</td></tr>`;
      }}
      tip.innerHTML = `<b>${{p.g}}</b> · %${{l}}<table>${{filas}}</table>`;
      tip.hidden = false;
      tip.style.left = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px";
      tip.style.top = (ev.clientY + 14) + "px";
    }});
    caza.addEventListener("mouseleave", () => {{ tip.hidden = true; cruz.setAttribute("visibility", "hidden"); }});
  }});
}}
dibuja();
</script>
</body></html>""", encoding="utf-8")


def _main():
    ap = argparse.ArgumentParser(description="Monitor como el Excel de planta: Outocal, modelo, Kalman y lab por guardia")
    ap.add_argument("--perfil", choices=list(PERFILES), required=True)
    ap.add_argument("--cada", type=float, default=0, help="minutos entre ciclos (0 = un solo ciclo)")
    ap.add_argument("--sin-extraer", action="store_true", help="no consulta el PI (usa el _CAL ya guardado)")
    args = ap.parse_args()
    monitor = MonitorExcel(PERFILES[args.perfil], extraer=not args.sin_extraer)
    while True:
        try:
            monitor.ciclo()
        except Exception as e:                           # un ciclo caído no tumba el monitor
            print(f"  CICLO FALLÓ: {type(e).__name__}: {str(e)[:500]}")
        if not args.cada:
            break
        time.sleep(args.cada * 60)


if __name__ == "__main__":
    _main()
