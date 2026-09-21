# ============================================================
# src/acquisition/from_pi_afsdk.py — Adquisición: intensidades desde PI OSIsoft (AF SDK directo, REFERENCIA)
# ============================================================
# RUTA VIGENTE DE EXTRACCIÓN: src/acquisition/from_pi.py (gateway HTTP desde
# WSL, ver data4cdpv1_local). Este archivo (AF SDK directo en Windows) queda
# como REFERENCIA/alternativa -misma fuente PI, mismos tags, transporte
# distinto-, útil si en algún momento no se tiene el gateway disponible pero
# sí una máquina Windows con PI AF Client Tools instalado.
#
# Servidor de PRODUCCIÓN 'tpi.southernperu.com.pe', tags del courier de cobre
# (idénticos a TAGS_COURIER en from_pi.py):
#
#   _296290_ConcFinal_CanalCu_ABB -> cu (n2cu)
#   _296290_ConcFinal_CanalFe_ABB -> fe (n1fe)
#   _296290_ConcFinal_CanalMo_ABB -> mo (n4mo)
#   _296290_ConcFinal_CanalSc_ABB -> sc (n6sc)
#   _296290_ConcFinal_CanalZn_ABB -> zn (n3zn)
#
# Fuente de ALTA FRECUENCIA (cada 15 min) del courier: el CSV que produce
# (Intensidades_<timestamp>.csv) es el mismo formato que produce
# src/acquisition/from_pi.py, y el que scripts/merge_cobre_data.py combina
# (--cobre-24) con la extracción de BAJA frecuencia de la BD (--cobre, ver
# src/acquisition/from_db.py) para obtener un dataset final más denso.
#
# REQUIERE WINDOWS: usa pythonnet (paquete `clr`) para cargar OSIsoft.AFSDK.dll
# (PI AF Client Tools instalado localmente). NO puede correr en este entorno
# Linux/WSL -> por eso la ruta vigente es from_pi.py vía el gateway desde WSL.
#
# FECHA_INICIO NO se cambia: es la fecha desde la que existen datos para estos
# tags en PI (2025-07-13 15:00:00) — confirmado por el usuario, no un supuesto.
#
# Uso (en la máquina Windows, con PI AF Client instalado):
#   python from_pi_afsdk.py
#   python from_pi_afsdk.py --fecha-fin "2026-09-21 00:00:00" --out Intensidades_nuevo.csv
# ============================================================

from datetime import datetime

import pandas as pd
import pytz

# --- Fecha desde la que existen datos para estos tags en PI: NO cambiar ---
FECHA_INICIO_FIJA = '2025-07-13 15:00:00'  # fecha de creación de las intensidades (courier)

SERVIDOR_NOMBRE_DEFAULT = 'tpi.southernperu.com.pe'
ZONA_HORARIA_LOCAL_DEFAULT = 'America/Lima'
INTERVALO_MUESTREO_DEFAULT = '15m'  # tiempo entre muestreo del concentrado final

# Tag PI -> nombre de columna cruda. Coincide 1:1 con el mapeo que ya aplica
# scripts/merge_cobre_data.py (mapping_24: fe->n1fe, cu->n2cu, zn->n3zn,
# mo->n4mo, sc->n6sc) -> no hace falta tocar ese script.
TAGS_CONFIG = {
    '_296290_ConcFinal_CanalCu_ABB': 'cu',
    '_296290_ConcFinal_CanalFe_ABB': 'fe',
    '_296290_ConcFinal_CanalMo_ABB': 'mo',
    '_296290_ConcFinal_CanalSc_ABB': 'sc',
    '_296290_ConcFinal_CanalZn_ABB': 'zn',
}


def extraer_con_afsdk(fecha_inicio: str = FECHA_INICIO_FIJA,
                      fecha_fin: str | None = None,
                      servidor_nombre: str = SERVIDOR_NOMBRE_DEFAULT,
                      zona_horaria_local: str = ZONA_HORARIA_LOCAL_DEFAULT,
                      intervalo_muestreo: str = INTERVALO_MUESTREO_DEFAULT,
                      tags_config: dict | None = None) -> pd.DataFrame:
    """Extrae del PI Data Archive (vía AF SDK) los tags del courier, interpolados
    cada `intervalo_muestreo`, y devuelve un DataFrame ancho (una columna por tag).

    Solo corre en Windows con PI AF Client Tools instalado (requiere el módulo
    `clr` de pythonnet + OSIsoft.AFSDK.dll).
    """
    import clr
    import sys as _sys
    _sys.path.append(r'C:\Program Files (x86)\PIPC\AF\PublicAssemblies\4.0')
    clr.AddReference('OSIsoft.AFSDK')

    from OSIsoft.AF.PI import PIServers, PIPoint
    from OSIsoft.AF.Time import AFTime, AFTimeSpan, AFTimeRange

    tags_config = tags_config or TAGS_CONFIG

    # --- 1. Configuración de tiempo ---
    tz = pytz.timezone(zona_horaria_local)
    inicio_dt = tz.localize(datetime.strptime(fecha_inicio, '%Y-%m-%d %H:%M:%S'))
    fin_dt = tz.localize(datetime.strptime(fecha_fin, '%Y-%m-%d %H:%M:%S')) if fecha_fin else datetime.now(tz)

    pi_inicio = AFTime(inicio_dt.isoformat())
    pi_fin = AFTime(fin_dt.isoformat())
    pi_intervalo = AFTimeSpan.Parse(intervalo_muestreo)

    # --- 2. Conexión ---
    pi_servers = PIServers()
    server = pi_servers[servidor_nombre]
    if not server.ConnectionInfo.IsConnected:
        server.Connect()

    print(f"Conectado a: {server.Name}")
    print(f"Rango: {fecha_inicio} hasta {fecha_fin if fecha_fin else 'Ahora'}")
    print(f"Intervalo: {intervalo_muestreo}")

    data_final = {}

    try:
        for tag_name, col_name in tags_config.items():
            try:
                point = PIPoint.FindPIPoint(server, tag_name)

                data_from_pi = point.InterpolatedValues(
                    AFTimeRange(pi_inicio, pi_fin),
                    pi_intervalo,
                    None,
                    False,
                )

                tag_values, timestamps = [], []
                for val in data_from_pi:
                    try:
                        timestamps.append(val.Timestamp.ToString("yyyy-MM-dd HH:mm:ss"))
                        v = val.Value
                        # Digital State (p.ej. 'No Data' o 'Error') -> NaN
                        tag_values.append(float('nan') if hasattr(v, 'Name') else float(v))
                    except Exception:
                        continue

                if timestamps:
                    data_final[col_name] = pd.Series(tag_values, index=pd.to_datetime(timestamps))
                    print(f"Extraído: {col_name} OK")
                else:
                    print(f"Aviso: {tag_name} no devolvió datos.")

            except Exception as e:
                print(f"Error en {tag_name}: {e}")

        if not data_final:
            raise RuntimeError("Ningún tag devolvió datos; revisar conexión/nombres de tags.")

        # --- 3. Consolidación ---
        df = pd.DataFrame(data_final)
        df.sort_index(inplace=True)
        df = df.ffill().bfill()               # relleno básico (mismo criterio que el script original)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    finally:
        server.Disconnect()


def _main():
    import argparse

    ap = argparse.ArgumentParser(description="Extrae intensidades del courier desde PI OSIsoft (AF SDK, Windows)")
    ap.add_argument("--fecha-inicio", default=FECHA_INICIO_FIJA,
                    help=f"NO cambiar salvo que sepas lo que haces: los tags solo tienen historia desde "
                         f"{FECHA_INICIO_FIJA}")
    ap.add_argument("--fecha-fin", default=None, help="'YYYY-MM-DD HH:MM:SS'; si se omite, usa 'ahora'")
    ap.add_argument("--servidor", default=SERVIDOR_NOMBRE_DEFAULT)
    ap.add_argument("--intervalo", default=INTERVALO_MUESTREO_DEFAULT)
    ap.add_argument("--out", default=None,
                    help="Default: Intensidades_<timestamp actual>.csv (mismo patrón que el script original)")
    args = ap.parse_args()

    if args.fecha_inicio != FECHA_INICIO_FIJA:
        print(f"AVISO: --fecha-inicio distinto del valor fijo conocido ({FECHA_INICIO_FIJA}); "
              f"si es anterior, es muy probable que no haya datos para esos tags antes de esa fecha.")

    df = extraer_con_afsdk(fecha_inicio=args.fecha_inicio, fecha_fin=args.fecha_fin,
                           servidor_nombre=args.servidor, intervalo_muestreo=args.intervalo)

    out = args.out or f"Intensidades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(out)
    print(f"\nProceso exitoso. Archivo generado: {out}")
    print("Cópialo a data/raw/ y pásalo como --cobre-24 a scripts/merge_cobre_data.py")


if __name__ == "__main__":
    _main()
