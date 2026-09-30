# data2TesisV2

Branch huérfano de `data2Tesis` (mismo repo remoto, `FitoCCF/data2Tesis`,
sin historia git compartida), clonado en su propia carpeta. Reescritura
completa del pipeline de `data2Tesis`, desacoplada (`src/pipeline/*.py`,
cada etapa con CLI propio + orquestador `src/run_pipeline.py`).

**Registro de investigación completo, con cifras medidas:**
`docs/bitacora_analisis.md`. Léelo antes de repetir un experimento — hay
seis pruebas ya cerradas con control negativo (insoluble, cabeza del
circuito, orden escalado/ortogonalización) que no se sostuvieron.

## Convenciones

- Dependencias con Pixi (`pixi run python notebooks/NN_*.py`), Python 3.14.
- `notebooks/NN_*.py` son wrappers delgados numerados en orden de ejecución
  del pipeline; la lógica real vive en `src/pipeline/`. Un módulo con nombre
  que empieza en dígito no es importable con `import`, pero sí con
  `__import__("NN_nombre")` (varios notebooks lo hacen para reusar funciones
  de otro sin duplicar código).
- `data/`, `reports/` y `models/*.joblib` están gitignorados por diseño —
  el comentario del propio `.gitignore` dice por qué: "no saturar GitHub
  ni filtrar info industrial". Si hace falta subir un CSV o una figura
  puntual, `git add -f`, pero no por defecto.

## Flujo del pipeline (revisado 2026-09-30, etapas 0-4)

```
00_getdata            crudos por fuente: courier_bd / courier_pi (recorded) / composito_pi
01a_limpieza_fuentes  PI -> una fila por lectura real; BD -> leyes 0 = NaN, duplicados unidos
01b_unificacion       tabla maestra: BD pegada al PI por VALOR de los 4 metales (fuente pi/ambos/bd)
01c_limpieza          periodo (train <= FECHA_CORTE_TRAIN / test / produccion), anomalia, valida, valida_ley
02_features           log-cocientes vs Cu (clustering) + fracciones y logSumI (regresión) + dil
03_escalado           corrección de dilución (solo Fe/Cu) + StandardScaler, ajustados en train
04_clusterizado       GMM K_GRUPOS=4 (Cu/Mo/Zn/Fe), diagnóstico de k, caracterización vs lab
```

Todo lo que se ajusta usa solo `periodo == 'train'`. Ninguna etapa borra
filas por una bandera: marca. Etapas 5-6, `run_pipeline.py` y el notebook 10
todavía usan la interfaz antigua (columnas `_ortho`, `limpiar()` viejo).

## Trampas ya encontradas

- **Crudos de la etapa 0: uno por fuente** (`courier_bd.csv` A,
  `courier_pi.csv` B, `composito_pi.csv` C), todos en hora local naive. A y B
  son el mismo instrumento: 108/126 muestras de la BD están con valores
  idénticos en el recorded del PI, a ~20 s de reloj — **no concatenarlas**
  (duplica lecturas) y **no interpolar el PI**. El courier da una lectura cada ~21 min
  y el PI la re-archiva cada 100 s (`ExcMax=100`): ~14 eventos por lectura,
  no un sensor congelado — la etapa 1a (`limpieza_fuentes.py`) las colapsa. El token del PI va
  en `PI_TOKEN` o `~/.pi_token`, nunca en el código.
- **Los tags PI `_296290_ConcFinal_*` ("C2 Courier cobre Concentrado Final")
  son el `sample_id=24` "Concentrado Colectivo" (equipo 5) de la BD**, no el
  `sample_id=10` "Concentrado Final" (equipo 1, otro courier). Verificado por
  valor el 2026-09-30: 126/126 muestras del 24 calzan con el PI; 0 de ~2,700
  de las otras 24 líneas de muestreo.
- **La etapa 5 (`modelos.py`, modelos locales por cluster) usa
  `KFold(shuffle=True)` sin corregir** — el mismo problema que ya se arregló
  en `data2Tesis` (`split_purgado`, walk-forward con purga y embargo) nunca
  se propagó aquí. Con el tamaño de muestra actual (~307-314), cualquier
  comparación de R² entre dos clusterings *necesita* un control aleatorio
  de partición (mismo tamaño de grupo, sin información real) o el resultado
  probablemente es ruido, no señal. Seis pruebas distintas lo confirmaron.
- **La ortogonalización contra `n6sc` se eliminó (2026-09-30).** Sus
  residuos arrastraban la deriva del instrumento (`n2cu_ortho` de +6,600 en
  2022H2 a -5,100 en 2026H2 con pCu casi constante). La etapa 2 es ahora
  `features.py`: log-cocientes contra Cu para clustering, fracciones de
  cierre + `logSumI` para regresión, sin artefacto. Si aparece una columna
  `_ortho` en un CSV viejo, es de antes de este cambio.
- **El courier no mide insolubles.** `a7a7`/`a6sol` en `works4cdp_assay`
  son una estimación del operador a partir de las demás leyes, no una
  lectura del sensor (confirmado con el equipo). `n7ech7` (el canal crudo
  real) está vacío para `sample_id=24` (Concentrado Colectivo). La única
  fuente de insoluble es el compósito de laboratorio vía PI (12h) o `pIns`
  en la BD (misma cadencia baja).
- **`n6sc` no mide el sólido de la muestra** (es el canal de dispersión):
  `corr(n6sc, pSol) = -0.15` sobre 344 muestras; `logSumI` sí (+0.57). `pSol` de
  alta frecuencia no existe en ninguna fuente revisada (ni compósito PI ni
  BD) — si algún día se instrumenta, usarlo como feature de sólido es la vía
  más prometedora que queda sin probar.

## Estado y decisiones
Ver `docs/estado.md` (sección `data2TesisV2`) y `docs/decisiones.md` en la
raíz de `Proyects/` para cómo esto se relaciona con `tesis_ucspv2/` — **el
encuadre de esta rama (solo Mo defendible) todavía no está reconciliado con
lo ya migrado ahí (4 elementos).**
