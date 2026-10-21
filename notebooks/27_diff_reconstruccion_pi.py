#!/usr/bin/env python3
# ============================================================
# CELDA 27 — DIFF: limpiar_pi() vs lecturas_por_evento()
# ============================================================
# Compara el reconstructor actual de la etapa 1a (lectura definida por los
# metales + n6sc pegado con ventana) contra el rediseño por evento
# (instantes de cambio agrupados por canal repetido). No escribe nada en
# data/: solo reporta el diff, para decidir el reemplazo con el número a la
# vista. Ver docs/bitacora_analisis.md §18.
#
#   pixi run python notebooks/27_diff_reconstruccion_pi.py
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from src.pipeline.config import RAW_COURIER_PI, CANALES
from src.pipeline.limpieza_fuentes import limpiar_pi, lecturas_por_evento

pd.set_option('display.width', 250)
crudo = pd.read_csv(RAW_COURIER_PI, parse_dates=['ts'])

viejo, r_v = limpiar_pi(crudo.copy())
nuevo, r_n = lecturas_por_evento(crudo.copy())

print('=' * 78)
print('RESUMEN DE CADA RECONSTRUCTOR')
print('=' * 78)
for nombre, r in [('limpiar_pi (actual)', r_v), ('lecturas_por_evento (nuevo)', r_n)]:
    print(f'\n{nombre}')
    for k, v in r.items():
        print(f'  {k}: {v}')

print('\n' + '=' * 78)
print('DIFF DE LECTURAS')
print('=' * 78)
print(f"  lecturas actual : {len(viejo)}")
print(f"  lecturas nuevo  : {len(nuevo)}   ({len(nuevo) - len(viejo):+d})")

# calce por VALOR: el nuevo fecha la lectura en el primer instante del evento,
# así que un Sc adelantado 3 s le cambia la hora sin cambiarle el contenido.
def llave(df):
    return df[CANALES].round(3).apply(lambda c: c.map(lambda v: 'na' if pd.isna(v) else f'{v:.3f}')).agg('|'.join, axis=1)

viejo = viejo.assign(k=llave(viejo))
nuevo = nuevo.assign(k=llave(nuevo))
if viejo['k'].duplicated().any() or nuevo['k'].duplicated().any():
    print(f"  (ojo: llaves repetidas  actual={int(viejo['k'].duplicated().sum())} nuevo={int(nuevo['k'].duplicated().sum())})")

m = viejo.merge(nuevo, on='k', how='outer', suffixes=('_v', '_n'), indicator=True)
solo_v = m[m['_merge'] == 'left_only']
solo_n = m[m['_merge'] == 'right_only']
ambos = m[m['_merge'] == 'both'].copy()
ambos['dt_s'] = (ambos['ts_n'] - ambos['ts_v']).dt.total_seconds()
print(f"  mismas 5 intensidades en ambos : {len(ambos)}")
print(f"  solo en actual                 : {len(solo_v)}")
print(f"  solo en nuevo                  : {len(solo_n)}")
print(f"\n  de las {len(ambos)} comunes, con la hora corrida: {int(ambos['dt_s'].ne(0).sum())}")
print('  distribucion del corrimiento (s, nuevo - actual):')
print(ambos.loc[ambos['dt_s'].ne(0), 'dt_s'].value_counts().sort_index().to_string())

if len(solo_v):
    print(f"\n  --- las {len(solo_v)} que SOLO estan en el actual ---")
    cols = ['ts_v'] + [c + '_v' for c in CANALES] + ['sostenida_s_v', 'n6sc_desfase_s_v']
    print(solo_v[cols].to_string(index=False))

if len(solo_n):
    print(f"\n  --- las {len(solo_n)} que SOLO estan en el nuevo ---")
    cols = ['ts_n'] + [c + '_n' for c in CANALES] + ['spread_s', 'canales', 'sostenida_s_n']
    print(solo_n[cols].to_string(index=False))

print('\n' + '=' * 78)
print('CHEQUEO FISICO: los 5 canales cambian juntos')
print('=' * 78)
print(f"  lecturas nuevas con menos de 5 canales: {int(nuevo['canales'].lt(5).sum())}")
print(f"  spread_s: p50={nuevo['spread_s'].median():.0f}  p99={nuevo['spread_s'].quantile(.99):.0f}  max={nuevo['spread_s'].max():.0f}")
dt = nuevo['ts'].diff().dt.total_seconds() / 60
print(f"  intervalo entre lecturas (min): p50={dt.median():.1f}  min={dt.min():.2f}")
print(f"  lecturas a < 2 min de la anterior: actual={int((viejo['ts'].diff().dt.total_seconds() < 120).sum())}  nuevo={int((dt * 60 < 120).sum())}")
