# -*- coding: utf-8 -*-
"""
pi_client.py
================================================================================
Cliente Python de la pasarela PiGateway (C# / AF SDK).

Permite consumir el PI Data Archive desde WSL2 Debian sin AF SDK ni pythonnet:
la pasarela corre en Windows 11 y este cliente habla con ella por HTTP/JSON.

Principios que respeta:
  - NO rellena, NO interpola y NO descarta valores malos. Devuelve el dato tal
    como lo entregó el historiador, con una máscara de calidad explícita.
  - Todos los timestamps del transporte son UTC. La conversión a America/Lima
    es una decisión del cliente, no del transporte.
  - El troceado temporal es responsabilidad del cliente: evita el error
    [-11091] "Event collection exceeded the maximum allowed" (ArcMaxCollect).

Uso mínimo:
    from pi_client import PiGateway

    pi = PiGateway()                       # autodetecta el host Windows
    print(pi.health())

    df = pi.recorded(['_294100_LIT_1011_ABB'], '2026-01-01', '2026-01-08')
    ancho = pi.to_wide(df)                 # pivote sin relleno

Requisitos: requests, pandas.
================================================================================
"""

from __future__ import annotations

import os
import re
import json
import time
import socket
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests


# =========================================================================== #
# Descubrimiento del host Windows desde WSL2
# =========================================================================== #
def _es_wsl() -> bool:
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except Exception:
        return False


def _candidatos_host() -> List[str]:
    """
    Direcciones donde puede estar el host Windows, en orden de preferencia.

    La IP del host cambia en cada reinicio de Windows bajo NAT, y en modo
    espejo (mirrored) el host responde en loopback. Probar varias evita
    depender de una configuración concreta.
    """
    cands: List[str] = []

    env = os.environ.get('PI_GATEWAY_HOST')
    if env:
        cands.append(env)

    if not _es_wsl():
        cands.append('127.0.0.1')
        return cands

    # Modo espejo: el host responde en loopback.
    cands.append('127.0.0.1')

    # Modo NAT: el host es el gateway por defecto.
    try:
        salida = subprocess.check_output(['ip', 'route'], text=True, timeout=5)
        m = re.search(r'^default via (\S+)', salida, re.M)
        if m:
            cands.append(m.group(1))
    except Exception:
        pass

    # El nameserver suele ser el mismo host, pero no siempre.
    try:
        with open('/etc/resolv.conf', 'r') as f:
            for linea in f:
                if linea.startswith('nameserver'):
                    cands.append(linea.split()[1].strip())
    except Exception:
        pass

    # WSL publica el host Windows como <hostname>.local
    try:
        cands.append(socket.gethostname() + '.local')
    except Exception:
        pass

    vistos, unicos = set(), []
    for c in cands:
        if c and c not in vistos:
            vistos.add(c)
            unicos.append(c)
    return unicos


def _probar_puerto(host: str, puerto: int, timeout: float = 1.0) -> bool:
    """Handshake TCP: distingue 'no hay nadie' de 'hay servicio'."""
    try:
        with socket.create_connection((host, puerto), timeout=timeout):
            return True
    except Exception:
        return False


def _host_windows(puerto: int = 5000, verbose: bool = False) -> str:
    """
    Primer candidato con el puerto abierto. Si ninguno responde, devuelve el
    primero para que el error posterior nombre una dirección concreta.
    """
    cands = _candidatos_host()
    for c in cands:
        if _probar_puerto(c, puerto):
            if verbose:
                print('[pi_client] pasarela detectada en {}:{}'.format(c, puerto))
            return c
    if verbose:
        print('[pi_client] ningún candidato respondió en el puerto {}: {}'.format(
            puerto, ', '.join(cands)))
    return cands[0] if cands else '127.0.0.1'


# =========================================================================== #
# Cliente
# =========================================================================== #
class PiGateway:
    """Cliente HTTP de la pasarela PiGateway."""

    def __init__(self,
                 host: Optional[str] = None,
                 puerto: int = 5000,
                 timeout: int = 600,
                 zona_local: str = 'America/Lima',
                 reintentos: int = 3,
                 espera_reintento: int = 10,
                 verbose: bool = True):
        self.host = host or _host_windows(puerto, verbose=verbose)
        self.puerto = puerto
        self.base = 'http://{}:{}'.format(self.host, self.puerto)
        self.timeout = timeout
        self.zona_local = zona_local
        self.reintentos = reintentos
        self.espera_reintento = espera_reintento
        self.verbose = verbose

        self.sesion = requests.Session()
        # requests descomprime gzip automáticamente; la pasarela lo respeta.
        self.sesion.headers.update({
            'Content-Type': 'application/json',
            'Accept-Encoding': 'gzip',
        })

    # ------------------------------------------------------------------ #
    # Infraestructura
    # ------------------------------------------------------------------ #
    def _log(self, msg: str) -> None:
        if self.verbose:
            print('[{}] {}'.format(datetime.now().strftime('%H:%M:%S'), msg),
                  flush=True)

    def _post(self, ruta: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base + ruta
        ultimo_error = None

        for intento in range(1, self.reintentos + 1):
            try:
                r = self.sesion.post(url, data=json.dumps(payload),
                                     timeout=self.timeout)
                if r.status_code >= 400:
                    detalle = r.text[:500]
                    raise RuntimeError('HTTP {} en {} -> {}'.format(
                        r.status_code, ruta, detalle))
                return r.json()
            except (requests.ConnectionError, requests.Timeout) as e:
                ultimo_error = e
                if intento < self.reintentos:
                    self._log('Intento {}/{} falló ({}). Reintento en {} s.'.format(
                        intento, self.reintentos, type(e).__name__,
                        self.espera_reintento))
                    time.sleep(self.espera_reintento)
            except Exception as e:
                raise

        raise ConnectionError(
            'No se pudo contactar la pasarela en {base}.\n'
            'Candidatos probados: {cands}\n'
            'Revise, en este orden:\n'
            '  1. En Windows: curl.exe http://localhost:{p}/health\n'
            '  2. En Windows: netstat -ano | findstr :{p}\n'
            '  3. En WSL:     curl -v --max-time 5 http://{host}:{p}/health\n'
            '  4. Si el firewall bloquea, use WSL en modo espejo '
            '(networkingMode=mirrored en %USERPROFILE%\\.wslconfig) y '
            'export PI_GATEWAY_HOST=127.0.0.1\n'
            'Error: {err}'.format(
                base=self.base, p=self.puerto, host=self.host,
                cands=', '.join(_candidatos_host()), err=ultimo_error))

    def health(self) -> Dict[str, Any]:
        r = self.sesion.get(self.base + '/health', timeout=30)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------ #
    # Conversión de la respuesta a DataFrame
    # ------------------------------------------------------------------ #
    def _a_df(self, data: Dict[str, Any], local: bool = True) -> pd.DataFrame:
        """
        Respuesta -> DataFrame largo: tag | t | value | good | digital

        Formato largo por diseño: el pivote a grilla ancha es una decisión
        posterior que exige elegir cómo alinear timestamps, y esa elección
        debe ser explícita.
        """
        filas: List[Dict[str, Any]] = []
        for tag, vals in data.items():
            if isinstance(vals, dict):          # /data con method=current
                vals = [vals]
            for v in vals or []:
                filas.append({
                    'tag': tag,
                    't': v.get('t'),
                    'value': v.get('v'),
                    'good': v.get('g'),
                    'digital': v.get('d'),
                    'error': v.get('e'),
                })

        df = pd.DataFrame(filas, columns=['tag', 't', 'value', 'good',
                                          'digital', 'error'])
        if df.empty:
            return df

        df['t'] = pd.to_datetime(df['t'], utc=True, format='ISO8601')
        if local:
            df['t'] = df['t'].dt.tz_convert(self.zona_local)
        return df.sort_values(['tag', 't']).reset_index(drop=True)

    @staticmethod
    def to_wide(df: pd.DataFrame, valor: str = 'value') -> pd.DataFrame:
        """
        Pivote a formato ancho SIN relleno: los huecos quedan como NaN
        explícitos. La imputación, si procede, es tarea del pipeline.
        """
        if df.empty:
            return df
        # dropna=False: una fila cuyos valores son todos malos DEBE seguir
        # existiendo en la grilla como NaN explícito, no desaparecer.
        return df.pivot_table(index='t', columns='tag', values=valor,
                              aggfunc='last', dropna=False)

    @staticmethod
    def mascara_calidad(df: pd.DataFrame) -> pd.DataFrame:
        """Pivote de la bandera IsGood, alineado con to_wide()."""
        if df.empty:
            return df
        return df.pivot_table(index='t', columns='tag', values='good',
                              aggfunc='last', dropna=False)

    # ------------------------------------------------------------------ #
    # Troceado temporal
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bloques(inicio: str, fin: str, dias: int):
        """Bloques semiabiertos [t0, t1): sin huecos ni traslapes."""
        t0 = pd.Timestamp(inicio)
        tf = pd.Timestamp(fin)
        while t0 < tf:
            t1 = min(t0 + pd.Timedelta(days=dias), tf)
            yield t0.strftime('%Y-%m-%d %H:%M:%S'), t1.strftime('%Y-%m-%d %H:%M:%S')
            t0 = t1

    def _serie_troceada(self, payload_base: Dict[str, Any],
                        inicio: str, fin: str,
                        chunk_dias: int) -> pd.DataFrame:
        partes = []
        bloques = list(self._bloques(inicio, fin, chunk_dias))
        for i, (t0, t1) in enumerate(bloques, 1):
            p = dict(payload_base)
            p['startTime'] = t0
            p['endTime'] = t1
            self._log('Bloque {}/{}: {} -> {}'.format(i, len(bloques), t0, t1))
            resp = self._post('/data', p)
            partes.append(self._a_df(resp.get('data', {})))
        if not partes:
            return pd.DataFrame()
        df = pd.concat(partes, ignore_index=True)
        return df.drop_duplicates(subset=['tag', 't']).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Métodos de extracción
    # ------------------------------------------------------------------ #
    def recorded(self, tags: Sequence[str], inicio: str, fin: str,
                 boundary: str = 'Inside',
                 filtro: str = '',
                 chunk_dias: int = 7) -> pd.DataFrame:
        """Dato crudo archivado. Base para auditoría, varianza e identificación."""
        return self._serie_troceada(
            {'method': 'recorded', 'tags': list(tags),
             'boundary': boundary, 'filterExpression': filtro},
            inicio, fin, chunk_dias)

    def interpolated(self, tags: Sequence[str], inicio: str, fin: str,
                     intervalo: str = '1m',
                     chunk_dias: int = 15) -> pd.DataFrame:
        """
        Grilla regular calculada por el servidor.

        Advertencia: un tag congelado y uno estable son indistinguibles aquí.
        Use recorded() para construir la máscara de calidad antes de confiar
        en esta salida.
        """
        return self._serie_troceada(
            {'method': 'interpolated', 'tags': list(tags), 'interval': intervalo},
            inicio, fin, chunk_dias)

    def plot(self, tags: Sequence[str], inicio: str, fin: str,
             intervalos: int = 640) -> pd.DataFrame:
        """Reducción para graficar. NUNCA para análisis ni modelado."""
        resp = self._post('/data', {
            'method': 'plot', 'tags': list(tags),
            'startTime': inicio, 'endTime': fin, 'intervals': intervalos})
        return self._a_df(resp.get('data', {}))

    def recorded_by_count(self, tags: Sequence[str], desde: str = '*',
                          count: int = 1, forward: bool = True) -> pd.DataFrame:
        """N eventos exactos desde una marca de tiempo, hacia adelante o atrás."""
        resp = self._post('/data', {
            'method': 'recorded_by_count', 'tags': list(tags),
            'startTime': desde, 'count': count, 'forward': forward})
        return self._a_df(resp.get('data', {}))

    def current(self, tags: Sequence[str]) -> pd.DataFrame:
        """Snapshot: valor en RAM del servidor, antes de la compresión."""
        resp = self._post('/data', {'method': 'current', 'tags': list(tags)})
        return self._a_df(resp.get('data', {}))

    def end_of_stream(self, tags: Sequence[str]) -> pd.DataFrame:
        """Último valor del stream. Requiere AF SDK 2.7 o superior."""
        resp = self._post('/data', {'method': 'end_of_stream', 'tags': list(tags)})
        return self._a_df(resp.get('data', {}))

    def summary(self, tags: Sequence[str], inicio: str, fin: str,
                duracion: str = '12h',
                tipos: Optional[Sequence[str]] = None,
                basis: str = 'TimeWeighted',
                chunk_dias: int = 90) -> pd.DataFrame:
        """
        Agregados calculados en el servidor.

        basis = 'TimeWeighted' es el promedio físicamente correcto sobre datos
        comprimidos. 'EventWeighted' sesga hacia los transitorios, que son los
        periodos con más eventos.

        Incluya siempre 'Count' y 'PercentGood': un bloque con muy pocos
        eventos delata un tag congelado.
        """
        tipos = list(tipos or ['Average', 'Minimum', 'Maximum', 'StdDev',
                               'Count', 'PercentGood'])
        partes = []
        for t0, t1 in self._bloques(inicio, fin, chunk_dias):
            self._log('Summary {} -> {}'.format(t0, t1))
            resp = self._post('/data', {
                'method': 'summary', 'tags': list(tags),
                'startTime': t0, 'endTime': t1,
                'summaryDuration': duracion, 'summaryTypes': tipos,
                'calculationBasis': basis})
            partes.append(self._a_df_summary(resp.get('data', {})))
        if not partes:
            return pd.DataFrame()
        df = pd.concat(partes, ignore_index=True)
        return df.drop_duplicates(subset=['tag', 'summary', 't']).reset_index(drop=True)

    def filtered_summary(self, tags: Sequence[str], inicio: str, fin: str,
                         filtro: str,
                         duracion: str = '12h',
                         tipos: Optional[Sequence[str]] = None,
                         basis: str = 'TimeWeighted') -> pd.DataFrame:
        """
        Agregados condicionados a una expresión de filtro evaluada en el
        servidor, por ejemplo: "'_294100_PP_007A_Speed_ABB' > 10".

        Úselo como verificación cruzada, no como sustituto de la
        reconstrucción duty/standby hecha en el cliente: el muestreo interno
        de la expresión no está bajo su control.
        """
        tipos = list(tipos or ['Average', 'Count', 'PercentGood'])
        resp = self._post('/data', {
            'method': 'filtered_summary', 'tags': list(tags),
            'startTime': inicio, 'endTime': fin,
            'summaryDuration': duracion, 'summaryTypes': tipos,
            'calculationBasis': basis, 'filterExpression': filtro})
        return self._a_df_summary(resp.get('data', {}))

    def _a_df_summary(self, data: Dict[str, Any]) -> pd.DataFrame:
        filas = []
        for tag, por_tipo in data.items():
            for tipo, vals in (por_tipo or {}).items():
                for v in vals or []:
                    filas.append({'tag': tag, 'summary': tipo, 't': v.get('t'),
                                  'value': v.get('v'), 'good': v.get('g')})
        df = pd.DataFrame(filas, columns=['tag', 'summary', 't', 'value', 'good'])
        if df.empty:
            return df
        df['t'] = pd.to_datetime(df['t'], utc=True, format='ISO8601') \
                    .dt.tz_convert(self.zona_local)
        return df.sort_values(['tag', 'summary', 't']).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Metadata y perfilado
    # ------------------------------------------------------------------ #
    def search(self, patron: str) -> pd.DataFrame:
        """Búsqueda de tags por patrón, por ejemplo *294100*."""
        resp = self._post('/search', {'pattern': patron})
        return pd.DataFrame(resp.get('points', []))

    def attributes(self, tags: Sequence[str],
                   atributos: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Metadata de puntos: step, compdev, archiving, creationdate, etc."""
        payload: Dict[str, Any] = {'tags': list(tags)}
        if atributos:
            payload['attributes'] = list(atributos)
        resp = self._post('/attributes', payload)
        filas = []
        for tag, attrs in resp.get('data', {}).items():
            fila = {'tag': tag}
            fila.update(attrs or {})
            filas.append(fila)
        return pd.DataFrame(filas)

    def perfil_historia(self, tags: Sequence[str],
                        origen: str = '1970-01-01') -> pd.DataFrame:
        """
        Perfila la historia realmente disponible por tag.

        Devuelve, por tag: primer evento grabado, si es el marcador
        "Pt Created", inicio útil de la serie, último evento y los atributos
        que deciden cómo reconstruirla.
        """
        resp = self._post('/firstevent', {'tags': list(tags), 'origin': origen})
        filas = []
        for tag, info in resp.get('data', {}).items():
            fila: Dict[str, Any] = {'tag': tag}
            if 'error' in info:
                fila['estado'] = 'ERROR: {}'.format(info['error'])
                filas.append(fila)
                continue

            primeros = info.get('first') or []
            ultimo = info.get('last') or []
            attrs = info.get('attributes') or {}

            if primeros:
                fila['primer_ts'] = primeros[0].get('t')
                v0 = primeros[0].get('d') or primeros[0].get('v')
                fila['primer_valor'] = v0
                fila['es_pt_created'] = (str(v0) == 'Pt Created')
                if len(primeros) > 1:
                    fila['segundo_ts'] = primeros[1].get('t')
                    fila['segundo_valor'] = (primeros[1].get('d')
                                             or primeros[1].get('v'))
                fila['estado'] = ('solo Pt Created (nunca recibió dato)'
                                  if fila['es_pt_created'] and len(primeros) < 2
                                  else 'ok')
            else:
                fila['estado'] = 'sin eventos grabados'

            if ultimo:
                fila['ultimo_ts'] = ultimo[0].get('t')
                fila['ultimo_valor'] = ultimo[0].get('d') or ultimo[0].get('v')

            for k in ('creationdate', 'archiving', 'compressing', 'step',
                      'compdev', 'compmax', 'excdev', 'excmax',
                      'pointtype', 'engunits'):
                fila[k] = attrs.get(k)
            filas.append(fila)

        df = pd.DataFrame(filas)
        if df.empty:
            return df

        # Inicio útil: si el primer evento es Pt Created, la serie empieza en el
        # siguiente evento.
        if 'es_pt_created' in df.columns:
            df['inicio_util'] = df.apply(
                lambda r: r.get('segundo_ts') if r.get('es_pt_created')
                else r.get('primer_ts'), axis=1)
        else:
            df['inicio_util'] = df.get('primer_ts')

        ini = pd.to_datetime(df['inicio_util'], utc=True,
                             format='ISO8601', errors='coerce')
        fin = pd.to_datetime(df.get('ultimo_ts'), utc=True,
                             format='ISO8601', errors='coerce')
        df['dias_historia'] = ((fin - ini).dt.total_seconds() / 86400.0).round(1)
        df['rezago_h'] = ((pd.Timestamp.now(tz='UTC') - fin)
                          .dt.total_seconds() / 3600.0).round(1)
        return df

    def ventana_comun(self, tags: Sequence[str]) -> Dict[str, Any]:
        """
        Ventana temporal común a todos los tags con dato: el rango defendible
        para un análisis conjunto.
        """
        df = self.perfil_historia(tags)
        ini = pd.to_datetime(df['inicio_util'], utc=True,
                             format='ISO8601', errors='coerce').dropna()
        fin = pd.to_datetime(df['ultimo_ts'], utc=True,
                             format='ISO8601', errors='coerce').dropna()
        if ini.empty or fin.empty:
            return {'inicio': None, 'fin': None, 'dias': 0, 'perfil': df}
        return {
            'inicio': ini.max(),
            'limita_inicio': df.loc[ini.idxmax(), 'tag'],
            'fin': fin.min(),
            'limita_fin': df.loc[fin.idxmin(), 'tag'],
            'dias': round((fin.min() - ini.max()).total_seconds() / 86400.0, 1),
            'perfil': df,
        }


# =========================================================================== #
# Prueba de humo
# =========================================================================== #
if __name__ == '__main__':
    pi = PiGateway()
    print('Pasarela:', pi.base)
    print('Health  :', pi.health())

    TAGS = ['_293200_Alim_Total_PB01_ABB',
            '_294100_LIT_1011_ABB',
            'C2_Sol_Overflow_Output']

    print('\n--- Perfil de historia ---')
    perfil = pi.perfil_historia(TAGS)
    print(perfil[['tag', 'inicio_util', 'ultimo_ts', 'dias_historia',
                  'step', 'archiving', 'estado']].to_string(index=False))

    print('\n--- Ventana común ---')
    v = pi.ventana_comun(TAGS)
    print('{} -> {}  ({} días)'.format(v['inicio'], v['fin'], v['dias']))

    print('\n--- Dato crudo, últimas 6 h ---')
    crudo = pi.recorded(TAGS[:1], '*-6h', '*')
    print(crudo.head().to_string(index=False))
    print('eventos: {} | buenos: {}'.format(len(crudo), int(crudo['good'].sum())))
