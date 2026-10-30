# Traspaso a `tesis_ucspv2` — conclusiones al 2026-10-29

Resumen de lo que `data2TesisV2` y la rama `AlimConc` dejan establecido, para llevarlo
a la tesis. Cada punto remite a la sección de `docs/bitacora_analisis.md` con las cifras.

## 1. Planteamiento

1. **Objetivo práctico:** reemplazar las ecuaciones del courier (Outocal), no al
   laboratorio. El Outocal es una regresión multivariable cuyo offset el operador corrige
   a mano cada día tras el compósito, sin criterio y con errores humanos (§24.1).
2. **Juez:** el compósito de 12 h del laboratorio, con el emparejamiento de planta
   (ventana de guardia `[t−1h, t+11h)`), y el Outocal `_CAL` como rival en las mismas
   guardias (§24.1, §24.3).
3. **El Kalman es la corrección de offset hecha bien:** automática, con ganancia
   estimada en train, sin intervención humana. Es la contribución que se puede defender
   frente a la práctica de planta (§24.1, §24.6).

## 2. Instrumento y datos

4. Los 5 canales del courier se miden juntos; el PI re-archiva cada 100 s y hay que
   reconstruir lecturas por evento (§18).
5. Noviembre de 2025 es un límite de instrumento (mantenimiento + recalibración): toda
   ventana va dentro de una época (§19).
6. La deriva de la magnitud tiene tres causas (dilución, escalón de recalibración,
   decaimiento de fuente −2.7 %/año); la invarianza a la dilución de `lr_zn_cu` es física
   del canal (§19).
7. Compósito y muestra puntual son dos cortes del mismo flujo, mismo laboratorio y método;
   el sesgo −1.2 de Cu entre ellos sigue sin explicación (§19.13).
8. **Cabeza:** dos líneas (rebose HC L1/L2); con alimentación > ~1450 t/h el DCS lava
   las líneas con agua y el courier lee agua (firma n1fe ≈ 350, n6sc ≈ 2,030). Cada
   línea sigue a su molino (L2 ↔ molino 2: 87 % de acierto). El compósito de cabeza sale
   de otro cortador y no se interrumpe; medido, eso **no** sesga hoy el error (§24.8).

## 3. Grupos de mineral

9. El concentrado **no tiene grupos**: tiene un gradiente de Mo (`lr_mo_cu` explica
   R² 0.67-0.77 sin ajustar nada). Las métricas internas (silhouette) premian el
   artefacto de dilución; solo la validación externa contra el compósito apunta bien (§19).
10. Los tipos de mineral se definen en la **cabeza** (k=2/3 sobre log-cocientes),
    validados por fase de mina, grupos del concentrado y recuperación (§15).
11. Toda comparación de particiones necesita control aleatorio. Con él, el modelo del
    concentrado en producción resultó **global** (`data2TesisV2`, ningún local pasa) o
    casi global (`AlimConc`, solo Mo-Mo1) (§23.1, §24.2).
12. El tipo de cabeza no mejora ni avisa en el concentrado; solo ayuda al Fe de cabeza
    (+0.07 de R²) (§23.3).

## 4. Resultados (31 días, guardias comunes, MAE contra compósito; §24.6)

| | Outocal | modelo | modelo + Kalman |
|---|---|---|---|
| Concentrado Cu | 0.928 | 0.976 | **0.713** |
| Concentrado Fe | 1.021 | 1.142 | **0.734** |
| Concentrado Mo (solo Mo k=4) | **0.111** | 0.164 | 0.146 |
| Cabeza Cu | **0.033** | 0.039 | 0.034 |
| Cabeza Fe | 0.181 | 0.189 | **0.166** |
| Cabeza Mo | **0.0033** | 0.0041 | 0.0034 |

13. Con Kalman el modelo le gana al Outocal en Cu y Fe del concentrado (−23 % y −28 %)
    y en Fe de cabeza. El Outocal gana el Mo del concentrado el último mes. Sin Kalman el
    modelo pierde en todo.
14. **Techo del Cu:** su R² bajo es falta de varianza (CV 7.5 %), y el residuo ya es ≤ el
    desacuerdo entre las dos referencias. Ninguna técnica probada lo mejora (§19.12,
    §24.4).
15. La ley de cabeza en línea es viable: en test le gana a la persistencia en las tres
    leyes (R² Cu 0.41 / Fe 0.60 / Mo 0.83 con Kalman) (§23.2).

## 5. Lo que no funcionó (resultados negativos, también son tesis)

16. Ortogonalización contra `n6sc`, insoluble, cabeza como feature del concentrado,
    offset de época, variantes de `dil`, tipo de cabeza como selector (§8-9, §16, §19.10,
    §23.3).
17. PCA semi-supervisado, GBM monotónico y GPR no aportan; fusionar con el Outocal está
    descartado por diseño (§24.4, §24.1).

## 6. Robustez operativa

18. **Cambios de mineral:** el modelo falló el Cu el 25-oct (−2.21) cuando entró
    mineral con más Zn y Mo; las relaciones débiles aprendidas en train se rompen fuera
    de rango, y el Kalman llega 12-24 h tarde. Registro de incidentes en §22 y §24.7.
    Hace falta un aviso de "mineral fuera de rango".
19. Atoros del courier y lavados por tonelaje producen ventanas sin datos: el monitor
    debe marcar "sin datos", nunca inventar un valor (§24.7, §24.8).

## 7. Abierto al 2026-10-29

- Prueba en ciego (desde el compósito del 28-oct 19:30): Fe con Kalman sin turno y Mo
  en log entrenado contra compósito. Decidir con 30-60 guardias (§24.5).
- Compósito del 29-oct 07:30 (salto de Mo a ~2.2, Cu ~23): verificar si el Cu fue real.
- Aviso de mineral fuera de rango; columna de cobertura de cabeza; causa de los lavados
  de L1 no explicados por tonelaje.
- Reconciliar con `tesis_ucspv2`: el encuadre "solo Mo defendible" de esta rama vs los
  4 elementos ya migrados. Con el juez nuevo (Outocal), Cu y Fe con Kalman **sí** son
  defendibles como reemplazo del Outocal, aunque su R² contra el laboratorio sea bajo.
