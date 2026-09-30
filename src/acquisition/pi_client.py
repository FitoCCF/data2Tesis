# -*- coding: utf-8 -*-
"""
pi_client.py — Cliente del PI System a través de la pasarela PiGateway.
================================================================================

Una sola dependencia real: ``requests``. ``pandas`` es opcional y se importa
solo cuando se pide un DataFrame, de modo que un backend web puede usar este
módulo sin arrastrar pandas a cada petición HTTP.

DÓNDE CORRE
--------------------------------------------------------------------------------
  - Laptop Linux (Slackware, Debian, la que sea), por WiFi o por cable.
  - Estación administrativa con Python, contra la pasarela de la red.
  - WSL2, para un backend Django que sirve a un frontend Vue.
  - Un pipeline de extracción masiva, desatendido, durante horas.

No requiere el AF SDK ni pythonnet: eso vive únicamente en la máquina que
ejecuta la pasarela.

DOS CAPAS DE API
--------------------------------------------------------------------------------
  Capa cruda      devuelve dict y list de Python puro. Es lo que quiere una
                  vista Django que va a serializar a JSON para Vue: sin
                  pandas de por medio, sin conversión de zona horaria, sin
                  coste de memoria. Métodos terminados en ``_raw``.

  Capa DataFrame  devuelve pandas.DataFrame con timestamps convertidos a la
                  zona local. Es lo que quiere un pipeline o un notebook.
                  Requiere pandas instalado.

Ambas hablan con los mismos endpoints; la segunda está construida sobre la
primera.

USO COMO BIBLIOTECA
--------------------------------------------------------------------------------
    from pi_client import PiGateway

    pi = PiGateway(host='10.25.18.85', puerto=5173, token='...')
    print(pi.health())

    # Vista Django -> JSON para Vue (sin pandas)
    datos = pi.datos_raw('recorded', tags=['_294100_LIT_1011_ABB'],
                         startTime='*-6h', endTime='*')

    # Pipeline -> DataFrame
    df = pi.recorded(['_294100_LIT_1011_ABB'], '2026-01-01', '2026-03-01')

USO EN DJANGO
--------------------------------------------------------------------------------
    # settings.py  (o variables de entorno del proceso)
    #   PI_GATEWAY_HOST=10.25.18.85
    #   PI_GATEWAY_PORT=5173
    #   PI_TOKEN=...
    #
    # views.py
    from pi_client import obtener_cliente
    pi = obtener_cliente()          # instancia única, segura entre hilos

``obtener_cliente()`` devuelve un singleton con sesiones HTTP por hilo, que
es lo que corresponde bajo un servidor WSGI con varios workers.

CONFIGURACIÓN POR ENTORNO
--------------------------------------------------------------------------------
    PI_GATEWAY_HOST     host de la pasarela (si falta, se autodetecta)
    PI_GATEWAY_PORT     puerto (por defecto 5000)
    PI_TOKEN            valor de la cabecera X-PI-Token
    PI_GATEWAY_TIMEOUT  segundos de espera por petición (por defecto 600)
    PI_ZONA             zona horaria de salida (por defecto America/Lima)

PRINCIPIOS QUE RESPETA
--------------------------------------------------------------------------------
  - NO rellena, NO imputa, NO descarta. Los huecos quedan como NaN
    explícitos y la calidad viaja en su propia estructura.
  - Los timestamps del transporte son UTC; la conversión a hora local ocurre
    al final y solo en la capa DataFrame.
  - El troceado temporal es del cliente: evita el error [-11091]
    (ArcMaxCollect) y los timeouts del servidor.
  - Registra por ``logging``, nunca por ``print``: en un backend, escribir a
    stdout desde una biblioteca ensucia los logs del servidor.
================================================================================
"""

from __future__ import annotations

import os
import re
import json
import time
import socket
import logging
import threading
import subprocess
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import requests

# pandas es OPCIONAL. Un backend Django que solo use la capa cruda no debe
# verse obligado a instalarlo ni a pagar su tiempo de importación.
try:
    import pandas as pd
except ImportError:          # pragma: no cover
    pd = None

__all__ = [
    'PiGateway', 'obtener_cliente', 'reiniciar_cliente',
    'PiError', 'PiConnError', 'PiAuthError', 'PiServerError', 'PiPandasError',
]

log = logging.getLogger(__name__)
# Evita el aviso "No handlers could be found" cuando el programa anfitrión no
# ha configurado logging. Una biblioteca no decide cómo se registran sus
# mensajes; solo evita romper por no haberlo decidido nadie.
log.addHandler(logging.NullHandler())


# =========================================================================== #
# Excepciones
# =========================================================================== #
class PiError(Exception):
    """Raíz de los errores del cliente. Permite capturar todo de una vez."""


class PiConnError(PiError):
    """No se pudo contactar la pasarela: caída, host equivocado, bloqueo."""


class PiAuthError(PiError):
    """Token ausente o incorrecto (HTTP 401/403)."""


class PiServerError(PiError):
    """La pasarela respondió con error. Suele originarse en el PI."""

    def __init__(self, mensaje: str, codigo: int = 0, tipo: str = ''):
        super().__init__(mensaje)
        self.codigo = codigo      # código HTTP
        self.tipo = tipo          # nombre de la excepción .NET, si la hubo

    @property
    def es_volumen(self) -> bool:
        """
        True si el error viene del límite ArcMaxCollect del PI, es decir,
        se pidieron demasiados eventos en una sola llamada. Se distingue
        de un timeout porque la solución es distinta: trocear más fino en
        lugar de esperar más.
        """
        return '-11091' in str(self) or 'exceeded the maximum' in str(self).lower()


class PiPandasError(PiError):
    """Se pidió un DataFrame y pandas no está instalado."""


def _exigir_pandas() -> None:
    if pd is None:
        raise PiPandasError(
            'Esta operación devuelve un DataFrame y pandas no está instalado. '
            'Instálelo (pip install pandas) o use el método *_raw equivalente, '
            'que devuelve dict y list de Python puro.')


# =========================================================================== #
# Descubrimiento del host de la pasarela
# =========================================================================== #
def _es_wsl() -> bool:
    """True si este proceso corre dentro de WSL."""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except Exception:
        return False


def _candidatos_host() -> List[str]:
    """
    Direcciones donde puede estar la pasarela, en orden de preferencia.

    Solo se usa cuando no se indicó host ni PI_GATEWAY_HOST. En una estación
    administrativa o en la laptop conviene indicarlo explícitamente; la
    autodetección está pensada para el caso WSL, donde la dirección del
    anfitrión Windows cambia entre reinicios bajo red NAT.
    """
    cands: List[str] = []

    env = os.environ.get('PI_GATEWAY_HOST')
    if env:
        cands.append(env)

    # Fuera de WSL no hay nada que deducir: la pasarela está en otra máquina
    # y su dirección debe darse. Se prueba loopback por si corre aquí mismo.
    if not _es_wsl():
        cands.append('127.0.0.1')
        return cands

    # Modo espejo de WSL: el anfitrión responde en loopback.
    cands.append('127.0.0.1')

    # Modo NAT: el anfitrión Windows es la puerta de enlace por defecto.
    try:
        salida = subprocess.check_output(['ip', 'route'], text=True, timeout=5)
        m = re.search(r'^default via (\S+)', salida, re.M)
        if m:
            cands.append(m.group(1))
    except Exception:
        pass

    # El servidor DNS suele ser el mismo anfitrión, aunque no siempre.
    try:
        with open('/etc/resolv.conf', 'r') as f:
            for linea in f:
                if linea.startswith('nameserver'):
                    cands.append(linea.split()[1].strip())
    except Exception:
        pass

    # Elimina repetidos conservando el orden de preferencia.
    vistos, unicos = set(), []
    for c in cands:
        if c and c not in vistos:
            vistos.add(c)
            unicos.append(c)
    return unicos


def _puerto_abierto(host: str, puerto: int, timeout: float = 1.0) -> bool:
    """Handshake TCP: distingue 'no hay nadie' de 'hay servicio'."""
    try:
        with socket.create_connection((host, puerto), timeout=timeout):
            return True
    except Exception:
        return False


def _detectar_host(puerto: int) -> str:
    """
    Primer candidato con el puerto abierto. Si ninguno responde, devuelve el
    primero, para que el error posterior nombre una dirección concreta en
    lugar de una lista.
    """
    cands = _candidatos_host()
    for c in cands:
        if _puerto_abierto(c, puerto):
            log.debug('Pasarela detectada en %s:%s', c, puerto)
            return c
    log.debug('Ningún candidato respondió en el puerto %s: %s', puerto, cands)
    return cands[0] if cands else '127.0.0.1'


# =========================================================================== #
# Cliente
# =========================================================================== #
class PiGateway:
    """
    Cliente HTTP de la pasarela PiGateway.

    Es seguro compartir una instancia entre hilos: cada hilo recibe su propia
    ``requests.Session``, porque Session no garantiza seguridad entre hilos
    y un servidor WSGI atiende peticiones en paralelo.
    """

    # ------------------------------------------------------------------ #
    # Construcción
    # ------------------------------------------------------------------ #
    def __init__(self,
                 host: Optional[str] = None,
                 puerto: Optional[int] = None,
                 token: Optional[str] = None,
                 timeout: Optional[int] = None,
                 zona_local: Optional[str] = None,
                 reintentos: int = 3,
                 espera_reintento: int = 10,
                 verbose: bool = False):
        """
        Todos los parámetros caen al entorno si se omiten, y del entorno a un
        valor por defecto. Ese orden permite que el mismo código funcione en
        un notebook (parámetros explícitos) y en Django (variables de
        entorno) sin ramificaciones.
        """
        self.puerto = int(puerto if puerto is not None
                          else os.environ.get('PI_GATEWAY_PORT', 5000))

        # El host es lo único que se autodetecta, porque es lo único que
        # cambia solo.
        self.host = host or os.environ.get('PI_GATEWAY_HOST') \
            or _detectar_host(self.puerto)

        self.base = 'http://{}:{}'.format(self.host, self.puerto)

        self.timeout = int(timeout if timeout is not None
                           else os.environ.get('PI_GATEWAY_TIMEOUT', 600))

        self.zona_local = zona_local or os.environ.get('PI_ZONA', 'America/Lima')

        # token=None -> se busca en el entorno. token='' -> sin token, de
        # forma deliberada. La distinción importa: permite desactivarlo desde
        # código aunque la variable de entorno exista.
        self.token = token if token is not None else os.environ.get('PI_TOKEN')

        self.reintentos = max(1, int(reintentos))
        self.espera_reintento = int(espera_reintento)

        # Compatibilidad: verbose=True eleva el nivel de este logger para que
        # un script de línea de comandos vea el progreso sin configurar nada.
        self.verbose = verbose
        if verbose and not log.handlers[1:]:
            logging.basicConfig(level=logging.INFO,
                                format='[%(asctime)s] %(message)s',
                                datefmt='%H:%M:%S')
            log.setLevel(logging.INFO)

        # Una sesión por hilo, guardada en almacenamiento local de hilo.
        self._local = threading.local()

    @classmethod
    def desde_entorno(cls, **kwargs) -> 'PiGateway':
        """Constructor explícito por entorno. Equivale a PiGateway()."""
        return cls(**kwargs)

    # ------------------------------------------------------------------ #
    # Sesión HTTP
    # ------------------------------------------------------------------ #
    @property
    def sesion(self) -> requests.Session:
        """
        Sesión del hilo actual, creada al primer uso.

        Reutilizar la sesión mantiene viva la conexión TCP entre peticiones,
        lo que importa cuando un pipeline hace cientos de llamadas seguidas.
        """
        s = getattr(self._local, 'sesion', None)
        if s is None:
            s = requests.Session()
            s.headers.update({
                'Content-Type': 'application/json',
                # requests descomprime gzip de forma transparente; la
                # pasarela solo comprime si ve esta cabecera.
                'Accept-Encoding': 'gzip',
                'User-Agent': 'pi_client/2.0',
            })
            if self.token:
                s.headers['X-PI-Token'] = self.token
            self._local.sesion = s
        return s

    def close(self) -> None:
        """Cierra la sesión de ESTE hilo. Las de otros hilos no se tocan."""
        s = getattr(self._local, 'sesion', None)
        if s is not None:
            try:
                s.close()
            except Exception:
                pass
            self._local.sesion = None

    def __enter__(self) -> 'PiGateway':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return '<PiGateway {} token={}>'.format(
            self.base, 'sí' if self.token else 'no')

    # ------------------------------------------------------------------ #
    # Transporte
    # ------------------------------------------------------------------ #
    def _revisar(self, r: requests.Response, ruta: str) -> Dict[str, Any]:
        """
        Traduce la respuesta HTTP a un dict, o a una excepción con contexto.
        """
        if r.status_code in (401, 403):
            raise PiAuthError(
                'La pasarela rechazó la petición por token (HTTP {}). Defina '
                'PI_TOKEN con el mismo valor que usa la pasarela, o pase '
                'token="..." al crear PiGateway.'.format(r.status_code))

        if r.status_code >= 400:
            detalle, tipo = r.text[:500], ''
            try:
                cuerpo = r.json()
                detalle = cuerpo.get('error', detalle)
                tipo = cuerpo.get('type', '')
            except Exception:
                # La pasarela devolvió algo que no es JSON: se conserva el
                # texto crudo, que suele bastar para diagnosticar.
                pass
            raise PiServerError('HTTP {} en {} -> {}'.format(
                r.status_code, ruta, detalle), codigo=r.status_code, tipo=tipo)

        try:
            return r.json()
        except ValueError as e:
            raise PiServerError(
                'Respuesta no es JSON válido en {}: {}'.format(ruta, e))

    def _post(self, ruta: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST con reintentos. Solo reintenta fallos de RED; un error devuelto
        por la pasarela es determinista y repetirlo no cambia nada.
        """
        url = self.base + ruta
        ultimo = None

        for intento in range(1, self.reintentos + 1):
            try:
                r = self.sesion.post(url, data=json.dumps(payload),
                                     timeout=self.timeout)
                return self._revisar(r, ruta)

            except (requests.ConnectionError, requests.Timeout) as e:
                ultimo = e
                # Una conexión rota deja la sesión en estado dudoso: se
                # descarta para que el siguiente intento abra una nueva.
                self.close()
                if intento < self.reintentos:
                    log.warning('Intento %s/%s falló (%s). Reintento en %s s.',
                                intento, self.reintentos, type(e).__name__,
                                self.espera_reintento)
                    time.sleep(self.espera_reintento)

        raise PiConnError(self._ayuda_conexion(ultimo))

    def _ayuda_conexion(self, error) -> str:
        """Mensaje de fallo con los pasos de diagnóstico concretos."""
        return (
            'No se pudo contactar la pasarela en {base}.\n'
            'Compruebe, en este orden:\n'
            '  1. Que gw4Pi.exe esté corriendo en la máquina con el AF SDK.\n'
            '  2. Desde esa máquina:  curl http://localhost:{p}/health\n'
            '  3. Desde aquí:         curl -v --max-time 5 {base}/health\n'
            '  4. Que el firewall permita el puerto {p}, y que el reenviador\n'
            '     esté activo si la regla está ligada a otro programa.\n'
            'Error: {err}'
        ).format(base=self.base, p=self.puerto, err=error)

    # ================================================================== #
    # CAPA CRUDA  —  dict y list de Python. Sin pandas.
    #                Es la que debe usar una vista Django.
    # ================================================================== #
    def health(self) -> Dict[str, Any]:
        """
        Estado de la pasarela: servidor, colectivo, miembro y timeouts.

        Útil como comprobación de vida en un backend, y como registro de
        trazabilidad al inicio de una extracción.
        """
        try:
            r = self.sesion.get(self.base + '/health', timeout=30)
        except (requests.ConnectionError, requests.Timeout) as e:
            self.close()
            raise PiConnError(self._ayuda_conexion(e))
        return self._revisar(r, '/health')

    def buscar_raw(self, patron: str) -> List[Dict[str, Any]]:
        """Busca tags por patrón. Devuelve una lista de dicts."""
        return self._post('/search', {'pattern': patron}).get('points', [])

    def atributos_raw(self, tags: Sequence[str],
                      atributos: Optional[Sequence[str]] = None
                      ) -> Dict[str, Dict[str, Any]]:
        """Metadata por tag: {tag: {atributo: valor}}."""
        payload: Dict[str, Any] = {'tags': list(tags)}
        if atributos:
            payload['attributes'] = list(atributos)
        return self._post('/attributes', payload).get('data', {})

    def datos_raw(self, metodo: str, tags: Sequence[str],
                  **parametros) -> Dict[str, Any]:
        """
        Acceso directo a /data, sin conversiones.

        Es el método que conviene en una vista Django: devuelve exactamente
        lo que emitió la pasarela, listo para serializar a JSON hacia Vue,
        con los timestamps en UTC.

            pi.datos_raw('recorded', tags=['TAG'],
                         startTime='*-6h', endTime='*')

            pi.datos_raw('summary', tags=['TAG'],
                         startTime='2026-01-01', endTime='2026-02-01',
                         summaryDuration='12h',
                         summaryTypes=['Average', 'Count'])
        """
        payload: Dict[str, Any] = {'method': metodo, 'tags': list(tags)}
        payload.update(parametros)
        return self._post('/data', payload).get('data', {})

    def perfil_raw(self, tags: Sequence[str],
                   origen: str = '1970-01-01') -> Dict[str, Any]:
        """Primer y último evento grabado por tag, más su metadata."""
        return self._post('/firstevent',
                          {'tags': list(tags), 'origin': origen}).get('data', {})

    # ================================================================== #
    # CONVERSIÓN A DataFrame
    # ================================================================== #
    def _a_df(self, data: Dict[str, Any], local: bool = True) -> 'pd.DataFrame':
        """
        Respuesta de /data -> DataFrame largo: tag | t | value | good | digital

        Formato largo por diseño. El pivote a grilla ancha exige decidir cómo
        alinear timestamps entre tags, y esa decisión debe ser explícita del
        pipeline, no un efecto secundario del transporte.
        """
        _exigir_pandas()
        filas: List[Dict[str, Any]] = []

        for tag, vals in data.items():
            # current y end_of_stream devuelven un dict suelto, no una lista.
            if isinstance(vals, dict):
                vals = [vals]
            for v in vals or []:
                filas.append({
                    'tag': tag,
                    't': v.get('t'),
                    'value': v.get('v'),
                    'good': v.get('g'),
                    'digital': v.get('d'),    # nombre del estado digital
                    'error': v.get('e'),      # estado del PI si el valor es malo
                })

        df = pd.DataFrame(filas, columns=['tag', 't', 'value', 'good',
                                          'digital', 'error'])
        if df.empty:
            return df

        df['t'] = pd.to_datetime(df['t'], utc=True, format='ISO8601')
        if local:
            df['t'] = df['t'].dt.tz_convert(self.zona_local)
        return df.sort_values(['tag', 't']).reset_index(drop=True)

    def _a_df_summary(self, data: Dict[str, Any]) -> 'pd.DataFrame':
        """Respuesta de summary -> tag | summary | t | value | good"""
        _exigir_pandas()
        filas = []
        for tag, por_tipo in data.items():
            for tipo, vals in (por_tipo or {}).items():
                for v in vals or []:
                    filas.append({'tag': tag, 'summary': tipo, 't': v.get('t'),
                                  'value': v.get('v'), 'good': v.get('g')})

        df = pd.DataFrame(filas, columns=['tag', 'summary', 't', 'value', 'good'])
        if df.empty:
            return df
        df['t'] = (pd.to_datetime(df['t'], utc=True, format='ISO8601')
                     .dt.tz_convert(self.zona_local))
        return df.sort_values(['tag', 'summary', 't']).reset_index(drop=True)

    @staticmethod
    def to_wide(df: 'pd.DataFrame', valor: str = 'value') -> 'pd.DataFrame':
        """
        Pivote a formato ancho SIN relleno: los huecos quedan como NaN
        explícitos. La imputación, si procede, es tarea del pipeline.

        dropna=False es deliberado: una fila cuyos valores son todos malos
        DEBE seguir existiendo en la grilla como NaN, no desaparecer.
        """
        _exigir_pandas()
        if df.empty:
            return df
        return df.pivot_table(index='t', columns='tag', values=valor,
                              aggfunc='last', dropna=False)

    @staticmethod
    def mascara_calidad(df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Pivote de la bandera IsGood, alineado con to_wide()."""
        _exigir_pandas()
        if df.empty:
            return df
        return df.pivot_table(index='t', columns='tag', values='good',
                              aggfunc='last', dropna=False)

    # ================================================================== #
    # Troceado temporal
    # ================================================================== #
    @staticmethod
    def _bloques(inicio: str, fin: str, dias: float):
        """
        Bloques semiabiertos [t0, t1): sin huecos ni traslapes.

        El troceado evita dos límites distintos del servidor: ArcMaxCollect,
        que acota el número de valores por llamada, y el OperationTimeOut del
        colectivo, que acota su duración.
        """
        _exigir_pandas()
        t0 = pd.Timestamp(inicio)
        tf = pd.Timestamp(fin)
        if t0 >= tf:
            raise ValueError('El inicio debe ser anterior al fin.')
        if dias <= 0:
            # Sin este piso, t1 = min(t0 + dias, tf) queda <= t0 y el bucle
            # de abajo no vuelve a satisfacer su condición de salida.
            raise ValueError('chunk_dias debe ser mayor que 0 (recibido: {}).'
                             .format(dias))
        # Tope defensivo: pandas.Timedelta desborda con valores enormes, y
        # este método se llama con "sin troceo" desde el extractor.
        dias = min(float(dias), 3650.0)
        while t0 < tf:
            t1 = min(t0 + pd.Timedelta(days=dias), tf)
            yield (t0.strftime('%Y-%m-%d %H:%M:%S'),
                   t1.strftime('%Y-%m-%d %H:%M:%S'))
            t0 = t1

    def _serie_troceada(self, payload_base: Dict[str, Any],
                        inicio: str, fin: str,
                        chunk_dias: float) -> 'pd.DataFrame':
        partes = []
        bloques = list(self._bloques(inicio, fin, chunk_dias))
        for i, (t0, t1) in enumerate(bloques, 1):
            p = dict(payload_base)
            p['startTime'] = t0
            p['endTime'] = t1
            if len(bloques) > 1:
                log.info('Bloque %s/%s: %s -> %s', i, len(bloques), t0, t1)
            resp = self._post('/data', p)
            partes.append(self._a_df(resp.get('data', {})))

        if not partes:
            return pd.DataFrame()
        df = pd.concat(partes, ignore_index=True)
        # Los bordes de bloques contiguos pueden repetir un evento.
        return df.drop_duplicates(subset=['tag', 't']).reset_index(drop=True)

    # ================================================================== #
    # CAPA DataFrame  —  métodos de extracción
    # ================================================================== #
    def search(self, patron: str) -> 'pd.DataFrame':
        """Búsqueda de tags por patrón, como DataFrame."""
        _exigir_pandas()
        return pd.DataFrame(self.buscar_raw(patron))

    def attributes(self, tags: Sequence[str],
                   atributos: Optional[Sequence[str]] = None) -> 'pd.DataFrame':
        """Metadata de puntos como DataFrame, una fila por tag."""
        _exigir_pandas()
        filas = []
        for tag, attrs in self.atributos_raw(tags, atributos).items():
            fila = {'tag': tag}
            fila.update(attrs or {})
            filas.append(fila)
        return pd.DataFrame(filas)

    def recorded(self, tags: Sequence[str], inicio: str, fin: str,
                 boundary: str = 'Inside',
                 filtro: str = '',
                 chunk_dias: float = 7) -> 'pd.DataFrame':
        """
        Dato crudo archivado.

        Es la base para auditoría, para medir la varianza real y para
        identificar dinámica. Un tag congelado y uno estable se distinguen
        aquí, y solo aquí: en una grilla interpolada son idénticos.
        """
        return self._serie_troceada(
            {'method': 'recorded', 'tags': list(tags),
             'boundary': boundary, 'filterExpression': filtro},
            inicio, fin, chunk_dias)

    def interpolated(self, tags: Sequence[str], inicio: str, fin: str,
                     intervalo: str = '1m',
                     chunk_dias: float = 15) -> 'pd.DataFrame':
        """
        Grilla regular calculada por el servidor.

        Advertencia: un tag congelado por una constante de respaldo del PLC y
        uno genuinamente estable producen exactamente la misma línea plana.
        Use recorded() para construir la máscara de calidad antes de confiar
        en esta salida.
        """
        return self._serie_troceada(
            {'method': 'interpolated', 'tags': list(tags), 'interval': intervalo},
            inicio, fin, chunk_dias)

    def plot(self, tags: Sequence[str], inicio: str, fin: str,
             intervalos: int = 640) -> 'pd.DataFrame':
        """
        Reducción para graficar. NUNCA para análisis ni modelado: el muestreo
        está sesgado hacia los extremos de cada intervalo por diseño.
        """
        return self._a_df(self.datos_raw(
            'plot', tags, startTime=inicio, endTime=fin, intervals=intervalos))

    def recorded_by_count(self, tags: Sequence[str], desde: str = '*',
                          count: int = 1, forward: bool = True) -> 'pd.DataFrame':
        """N eventos exactos desde una marca de tiempo, hacia adelante o atrás."""
        return self._a_df(self.datos_raw(
            'recorded_by_count', tags,
            startTime=desde, count=count, forward=forward))

    def current(self, tags: Sequence[str]) -> 'pd.DataFrame':
        """Snapshot: valor vigente en la RAM del servidor, antes de archivarse."""
        return self._a_df(self.datos_raw('current', tags))

    def end_of_stream(self, tags: Sequence[str]) -> 'pd.DataFrame':
        """Último valor del stream. Requiere AF SDK 2.7 o superior."""
        return self._a_df(self.datos_raw('end_of_stream', tags))

    def summary(self, tags: Sequence[str], inicio: str, fin: str,
                duracion: str = '12h',
                tipos: Optional[Sequence[str]] = None,
                basis: str = 'TimeWeighted',
                chunk_dias: float = 90) -> 'pd.DataFrame':
        """
        Agregados calculados en el servidor.

        basis='TimeWeighted' es el promedio físicamente correcto sobre datos
        comprimidos. 'EventWeighted' sesga el resultado hacia los periodos con
        más eventos, es decir, hacia los transitorios.

        Incluya siempre 'Count' y 'PercentGood': un bloque de 12 h con uno o
        dos eventos en un tag de flujo delata un congelamiento, y sin Count
        eso no se ve.

        Cuidado con 'Total': el totalizador del PI asume tasas por DÍA. Si el
        tag está en m3/h, el total sale 24 veces menor salvo que se aplique
        el factor de conversión.
        """
        tipos = list(tipos or ['Average', 'Minimum', 'Maximum', 'StdDev',
                               'Count', 'PercentGood'])
        partes = []
        for t0, t1 in self._bloques(inicio, fin, chunk_dias):
            log.info('Summary %s -> %s', t0, t1)
            data = self.datos_raw('summary', tags,
                                  startTime=t0, endTime=t1,
                                  summaryDuration=duracion,
                                  summaryTypes=tipos,
                                  calculationBasis=basis)
            partes.append(self._a_df_summary(data))

        if not partes:
            return pd.DataFrame()
        df = pd.concat(partes, ignore_index=True)
        return df.drop_duplicates(
            subset=['tag', 'summary', 't']).reset_index(drop=True)

    def filtered_summary(self, tags: Sequence[str], inicio: str, fin: str,
                         filtro: str,
                         duracion: str = '12h',
                         tipos: Optional[Sequence[str]] = None,
                         basis: str = 'TimeWeighted') -> 'pd.DataFrame':
        """
        Agregados condicionados a una expresión evaluada en el servidor,
        por ejemplo: "'_294100_PP_007A_Speed_ABB' > 10".

        Conviene usarlo como verificación cruzada, no como sustituto de una
        reconstrucción hecha en el cliente: el muestreo interno de la
        expresión no está bajo su control.
        """
        tipos = list(tipos or ['Average', 'Count', 'PercentGood'])
        data = self.datos_raw('filtered_summary', tags,
                              startTime=inicio, endTime=fin,
                              summaryDuration=duracion,
                              summaryTypes=tipos,
                              calculationBasis=basis,
                              filterExpression=filtro)
        return self._a_df_summary(data)

    # ================================================================== #
    # Perfilado de historia
    # ================================================================== #
    def perfil_historia(self, tags: Sequence[str],
                        origen: str = '1970-01-01') -> 'pd.DataFrame':
        """
        Historia realmente disponible por tag.

        Devuelve el primer evento grabado, si ese primer evento es el
        marcador "Pt Created", el inicio útil de la serie, el último evento y
        los atributos que deciden cómo reconstruirla.

        El inicio útil NO es la fecha de creación del tag: si hay archivos
        .arc desregistrados, la historia disponible empieza después.
        """
        _exigir_pandas()
        filas = []

        for tag, info in self.perfil_raw(tags, origen).items():
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
                # En un valor digital el nombre está en 'd'; en uno numérico
                # el dato está en 'v'.
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

        # Inicio útil: si el primer evento es el marcador de creación, la
        # serie de proceso empieza en el siguiente.
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
        # Horas desde el último evento: delata un tag muerto o congelado.
        df['rezago_h'] = ((pd.Timestamp.now(tz='UTC') - fin)
                          .dt.total_seconds() / 3600.0).round(1)
        return df

    def ventana_comun(self, tags: Sequence[str]) -> Dict[str, Any]:
        """
        Ventana temporal común a todos los tags con dato: el rango defendible
        para un análisis conjunto. Indica además qué tag impone cada extremo.
        """
        _exigir_pandas()
        df = self.perfil_historia(tags)
        if df.empty:
            return {'inicio': None, 'fin': None, 'dias': 0, 'perfil': df}

        ini = pd.to_datetime(df['inicio_util'], utc=True,
                             format='ISO8601', errors='coerce').dropna()
        fin = pd.to_datetime(df.get('ultimo_ts'), utc=True,
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
# Instancia compartida, para Django y otros procesos de larga vida
# =========================================================================== #
_cliente: Optional[PiGateway] = None
_candado = threading.Lock()


def obtener_cliente(**kwargs) -> PiGateway:
    """
    Devuelve una instancia única de PiGateway, creada al primer uso.

    Pensado para un backend: crear un cliente por petición desperdicia la
    conexión persistente y la autodetección de host. La instancia es segura
    entre hilos porque cada uno usa su propia sesión HTTP.

        from pi_client import obtener_cliente
        pi = obtener_cliente()

    Los kwargs solo se aplican en la creación; llamadas posteriores
    devuelven la instancia existente. Use reiniciar_cliente() para cambiar
    la configuración.
    """
    global _cliente
    if _cliente is None:
        with _candado:
            # Doble comprobación: otro hilo pudo crearlo mientras este
            # esperaba el candado.
            if _cliente is None:
                _cliente = PiGateway(**kwargs)
                log.info('Cliente PI inicializado: %s', _cliente.base)
    return _cliente


def reiniciar_cliente(**kwargs) -> PiGateway:
    """Descarta la instancia compartida y crea otra con nueva configuración."""
    global _cliente
    with _candado:
        if _cliente is not None:
            _cliente.close()
        _cliente = PiGateway(**kwargs)
    return _cliente


# =========================================================================== #
# MODO STANDALONE
# ---------------------------------------------------------------------------
# Este bloque solo se ejecuta al invocar el archivo directamente. Al
# importarlo como biblioteca —desde un pipeline, desde Django, desde un
# notebook— nada de esto corre.
#
# Es un CLI de DIAGNÓSTICO: comprobar que la pasarela responde, mirar unos
# tags, ver desde cuándo hay historia. Para extracción masiva a CSV está
# pi_tool.py, que usa esta misma biblioteca.
#
#   python3 pi_client.py                          estado y autodiagnóstico
#   python3 pi_client.py health
#   python3 pi_client.py buscar "*294100*"
#   python3 pi_client.py atributos TAG1,TAG2
#   python3 pi_client.py perfil TAG1,TAG2
#   python3 pi_client.py crudo TAG "*-6h" "*"
#   python3 pi_client.py actual TAG1,TAG2
#
# Conexión: por entorno (PI_GATEWAY_HOST, PI_GATEWAY_PORT, PI_TOKEN) o por
# opciones --host, --puerto, --token, que tienen prioridad.
#
#   python3 pi_client.py --host 10.25.18.85 --puerto 5173 health
# =========================================================================== #
def _tabla(filas: List[Dict[str, Any]], columnas: Sequence[str]) -> str:
    """
    Tabla de ancho fijo sin pandas.

    El modo standalone debe funcionar en una estación donde solo haya
    requests instalado, así que no puede depender de DataFrame.to_string().
    """
    if not filas:
        return '  (sin resultados)'
    anchos = {c: len(c) for c in columnas}
    for f in filas:
        for c in columnas:
            anchos[c] = max(anchos[c], len(str(f.get(c, '') or '')))
    lineas = ['  ' + '  '.join(c.ljust(anchos[c]) for c in columnas)]
    lineas.append('  ' + '  '.join('-' * anchos[c] for c in columnas))
    for f in filas:
        lineas.append('  ' + '  '.join(
            str(f.get(c, '') or '').ljust(anchos[c]) for c in columnas))
    return '\n'.join(lineas)


def _main(argv: List[str]) -> int:
    # Se extraen primero las opciones con --, y lo que quede son los
    # argumentos posicionales del subcomando.
    opciones: Dict[str, str] = {}
    posicionales: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith('--'):
            if '=' in a:
                k, v = a[2:].split('=', 1)
            else:
                k = a[2:]
                v = argv[i + 1] if i + 1 < len(argv) else ''
                i += 1
            opciones[k] = v
        else:
            posicionales.append(a)
        i += 1

    if opciones.get('help') is not None or '-h' in argv:
        print(__doc__.split('USO COMO BIBLIOTECA')[0])
        return 0

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    pi = PiGateway(
        host=opciones.get('host'),
        puerto=int(opciones['puerto']) if 'puerto' in opciones else None,
        token=opciones.get('token'))

    comando = posicionales[0] if posicionales else 'health'
    # Los tags se aceptan separados por comas: cómodo para escribir a mano.
    tags = [t.strip() for t in posicionales[1].split(',')] \
        if len(posicionales) > 1 else []

    print('Pasarela: {}'.format(pi.base))

    # Toda la conexión va dentro de un try: los errores del cliente son
    # tipados y merecen un mensaje distinto cada uno.
    try:
        h = pi.health()
    except PiAuthError as e:
        print('\n[token] {}'.format(e))
        return 2
    except PiConnError as e:
        print('\n[conexión] {}'.format(e))
        return 1

    print('PI: {} | colectivo: {} ({}) | SDK {} | timeout {} s'.format(
        h.get('server'), h.get('collective'), h.get('memberType'),
        h.get('sdk'), h.get('operationTimeoutS')))

    try:
        # ---- health: ya está impreso arriba; añade el autodiagnóstico ----
        if comando == 'health':
            if not posicionales:
                print('\nMétodos disponibles: {}'.format(
                    ', '.join(h.get('methods', []))))
                print('pandas: {}'.format(
                    'disponible' if pd is not None else 'NO instalado '
                    '(la capa DataFrame no funcionará)'))
                print('\nSubcomandos: health | buscar | atributos | perfil | '
                      'crudo | actual')
                print('Para extracción masiva a CSV: pi_tool.py')
            return 0

        # ---- buscar -------------------------------------------------------
        if comando == 'buscar':
            if len(posicionales) < 2:
                print('Falta el patrón. Ej: buscar "*294100*"')
                return 1
            puntos = pi.buscar_raw(posicionales[1])
            print('\n{} tags encontrados:'.format(len(puntos)))
            print(_tabla(puntos, ['tag', 'descriptor', 'instrumenttag', 'engunits']))
            return 0

        # ---- atributos ----------------------------------------------------
        if comando == 'atributos':
            if not tags:
                print('Faltan los tags.')
                return 1
            data = pi.atributos_raw(tags)
            for tag, attrs in data.items():
                print('\n{}'.format(tag))
                for k in sorted(attrs or {}):
                    print('  {:<18} {}'.format(k, attrs[k]))
            return 0

        # ---- perfil -------------------------------------------------------
        if comando == 'perfil':
            if not tags:
                print('Faltan los tags.')
                return 1
            filas = []
            for tag, info in pi.perfil_raw(tags).items():
                f = {'tag': tag}
                if 'error' in info:
                    f['estado'] = 'ERROR: {}'.format(info['error'])
                    filas.append(f)
                    continue
                primeros = info.get('first') or []
                ultimo = info.get('last') or []
                attrs = info.get('attributes') or {}
                if primeros:
                    v0 = primeros[0].get('d') or primeros[0].get('v')
                    # Si el primer evento es el marcador de creación, la serie
                    # de proceso empieza en el siguiente.
                    pt_created = str(v0) == 'Pt Created'
                    f['inicio_util'] = (primeros[1].get('t')
                                        if pt_created and len(primeros) > 1
                                        else primeros[0].get('t'))
                    f['estado'] = ('solo Pt Created' if pt_created and
                                   len(primeros) < 2 else 'ok')
                else:
                    f['estado'] = 'sin eventos'
                f['ultimo'] = ultimo[0].get('t') if ultimo else ''
                f['step'] = attrs.get('step')
                f['archiving'] = attrs.get('archiving')
                f['compdev'] = attrs.get('compdev')
                filas.append(f)
            print('\nHistoria disponible (timestamps en UTC):')
            print(_tabla(filas, ['tag', 'inicio_util', 'ultimo', 'step',
                                 'archiving', 'compdev', 'estado']))
            return 0

        # ---- crudo --------------------------------------------------------
        if comando == 'crudo':
            if not tags:
                print('Faltan los tags.')
                return 1
            desde = posicionales[2] if len(posicionales) > 2 else '*-6h'
            hasta = posicionales[3] if len(posicionales) > 3 else '*'
            data = pi.datos_raw('recorded', tags,
                                startTime=desde, endTime=hasta)
            for tag, vals in data.items():
                buenos = sum(1 for v in vals if v.get('g'))
                print('\n{}: {} eventos, {} buenos, {} malos'.format(
                    tag, len(vals), buenos, len(vals) - buenos))
                # Primeros y últimos: suficiente para ver si el tag se mueve.
                muestra = vals[:3] + (['...'] if len(vals) > 6 else []) + vals[-3:]
                for v in muestra:
                    if v == '...':
                        print('    ...')
                    else:
                        print('    {}  {}  good={}'.format(
                            v.get('t'), v.get('d') or v.get('v'), v.get('g')))
            return 0

        # ---- actual -------------------------------------------------------
        if comando == 'actual':
            if not tags:
                print('Faltan los tags.')
                return 1
            data = pi.datos_raw('current', tags)
            filas = [{'tag': t,
                      'timestamp_utc': v.get('t'),
                      'valor': v.get('d') or v.get('v'),
                      'good': v.get('g')}
                     for t, v in data.items()]
            print('\nSnapshot:')
            print(_tabla(filas, ['tag', 'timestamp_utc', 'valor', 'good']))
            return 0

        print('\nSubcomando desconocido: {}'.format(comando))
        return 1

    except PiAuthError as e:
        print('\n[token] {}'.format(e))
        return 2
    except PiServerError as e:
        print('\n[servidor] {}'.format(e))
        # es_volumen distingue el límite ArcMaxCollect de un timeout: la
        # solución es trocear más fino, no esperar más.
        if e.es_volumen:
            print('Pida un rango más corto: se excedió el límite de eventos '
                  'por llamada del PI (ArcMaxCollect).')
        return 3
    except PiError as e:
        print('\n[{}] {}'.format(type(e).__name__, e))
        return 1
    finally:
        pi.close()


if __name__ == '__main__':
    import sys as _sys
    raise SystemExit(_main(_sys.argv[1:]))
