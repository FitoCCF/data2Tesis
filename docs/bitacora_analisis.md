# Bitácora de análisis — diagnóstico y recalibración del pipeline

Registro del razonamiento, las mediciones y las decisiones tomadas durante la
sesión de diagnóstico. Sirve para retomar el trabajo en otra máquina sin repetir
los experimentos. **Todos los números de este documento están medidos**, no
estimados; cada uno indica el protocolo con que se obtuvo.

Última actualización: 2026-09-25

---

## 0. Cómo retomar

```bash
git clone git@github.com-ucsp:FitoCCF/data2Tesis.git
cd data2Tesis && git checkout data2TesisV2
pixi install
```

**Los datos NO están en el repositorio** (`.gitignore` excluye `data/`, `reports/`
y `models/*.joblib`). Hay que copiarlos por medio privado. Sin ellos el código
corre pero no hay nada que procesar.

Archivos mínimos para reproducir:

| Archivo | Qué es |
|---|---|
| `data/processed/intensidades_cobre.csv` | stream del analizador, ~40k lecturas |
| `data/raw/assay_lab_courier_pi.csv` | compósito 12 h, 2590 ensayos (referencia de validación) |
| `data/processed/intensidad_cobre_24_completo_filtrado.csv` | 314 muestras puntuales de laboratorio |
| `data/raw/assay_lab_AIQ_pi.csv` | salida de la calibración de fábrica del analizador |

Luego:

```bash
pixi run python notebooks/10_recalibracion_composito.py   # regenera todos los .joblib
pixi run python notebooks/11_figuras_resultados.py        # regenera figuras y tabla
```

---

## 1. Estado actual: qué funciona y qué no

Evaluación contra el compósito de 12 h, n = 228 ventanas, abr–ago 2026, fuera de muestra.

| elemento | corr | R² | MAE | pendiente | fábrica | techo | veredicto |
|---|---|---|---|---|---|---|---|
| **Mo** | 0.953 | 0.906 | 0.157% | 0.880 | 0.948 | 0.961 | Funciona. En el techo físico |
| **Fe** | 0.508 | 0.244 | 0.914% | 0.317 | 0.470 | 0.743 | Supera a fábrica, no operativo |
| **Cu** | 0.484 | 0.086 | 1.390% | 0.268 | 0.517 | 0.634 | No funciona |
| **Zn** | — | — | — | — | — | — | Sin validar (el compósito no trae Zn) |

`fábrica` = calibración del fabricante del analizador. `techo` = correlación máxima
alcanzable, por colocación triple.

**Lectura operativa de la pendiente**: si el Fe real sube 1 punto, el modelo sube
0.32. Para control de proceso eso no sirve. Mo con 0.88 sí.

### Veredicto por etapa

| etapa | veredicto | evidencia |
|---|---|---|
| 0 adquisición | OK | — |
| 1 filtros deterministas | OK | 8.5% de rechazo legítimo |
| 1 IsolationForest | OK tras el arreglo | 31% → 2.7% de rechazo |
| 1 sensor congelado | **ROTO** | rechaza 6.6% por el forward-fill de PI |
| 2 ortogonalización | **DECORATIVA** | no-op algebraico + premisa falsa |
| 3 escalado | sin valor propio | solo alimenta la etapa 4 |
| 4 clustering | **NO APORTA** | +0.001 de correlación |
| 5 parte global | OK | es lo que produce las estimaciones |
| 5 parte local | **NO APORTA** | +0.001 |
| 6 inferencia | OK | integración probada |
| ruta B (Fe) | OK | 0.278 → 0.508 |

**El pipeline que funciona es: filtros → features → una regresión global.**

---

## 2. Hallazgos principales, en orden de importancia

### 2.1 Techo de información por colocación triple

Con tres estimaciones del mismo material (modelo / calibración de fábrica /
compósito) se despeja el error de cada una sin conocer la verdad. n = 184.

```
el   techo T~comp  fabrica~T  modelo~T | fabrica~comp  modelo~comp   margen
cu          0.634      0.815     0.798          0.517        0.505    0.128
fe          0.743      0.633     0.374          0.470        0.278    0.464
mo          0.961      0.986     0.996          0.948        0.958    0.003
```

Supuesto: errores mutuamente independientes. Modelo y fábrica comparten las
mismas mediciones XRF, así que la correlación entre ellos está inflada y los
techos quedan **subestimados** (son cotas conservadoras).

Validado por el usuario: el compósito lo corta personal de metalurgia justo antes
de la etapa de rayos X, sin proceso intermedio. Mismo material.

### 2.2 Deriva del instrumento: −22% anual

```
periodo   cu_lab  fe_lab    n2cu     n1fe    n6sc
2025Q3     26.15   27.28   39456    37122    1714
2026Q3     24.21   28.77   30929    28291    1365
```

Las intensidades caen 22–25% mientras las leyes reales casi no se mueven.
Sensibilidad `n1fe/pFe`: 1411 (2022) → 1222 (2026).

**Consecuencia**: la correlación intra-trimestre es MAYOR que la agrupada de 4
años. Para `fe_cu` vs Cu la agrupada (−0.207) es peor que cualquier trimestre
individual (−0.22 a −0.50). Agrupar el histórico mete la deriva como confusora.

### 2.3 Las fracciones de cierre son inmunes a la deriva

```
deriva relativa entre trimestres (2025Q3–2026Q3):
  n2cu_f   1.2%     n1fe_f   1.3%     n4mo_f   9.5%
  n3zn_f  17.6%     SumI/n6sc 17.4%
  n2cu    22.8%     n1fe    24.1%     n6sc    25.1%
```

`f_i = I_i / Σ metales` cancela la ganancia común del decaimiento de fuente.
`n6sc` NO decae al mismo ritmo, por eso normalizar por él reintroduce deriva.

### 2.4 `n6sc` no mide el sólido

Contra 309 valores de sólido medidos en laboratorio:

```
corr(n6sc,     pSol) = -0.065      <- esencialmente cero
corr(SumI,     pSol) = +0.659      <- este sí
corr(log SumI, pSol) = +0.655
```

La etapa 2 se llama "corrección por dilución (n6sc)" y su docstring afirma que
n6sc es "el sólido en la muestra / dilución". **Esa premisa es falsa.** No cambia
ningún número (la ortogonalización es un no-op algebraico de todos modos) pero la
justificación escrita en la tesis está mal.

Lo que sí es proxy del sólido es `SumI`, lo que valida retroactivamente `logSumI`
como feature de la ruta B, elegida por backtest sin saber por qué funcionaba.

### 2.5 La ortogonalización es un no-op algebraico

`m_ortho = m − a·n6sc − b`, y `n6sc` está en `FEATS_REGRESION`. Entonces
`span{m_ortho, n6sc} = span{m, n6sc}`. Para Ridge son idénticos:

```
pFe: A(4 ortho + n6sc) R2=0.351  |  B(4 crudas + n6sc) R2=0.352
pCu: A R2=0.690                  |  B R2=0.690
```

Solo afecta al clustering.

### 2.6 Régimen de mineral: el efecto es real, el instrumento no lo ve

**Ruteando por insolubles REALES de laboratorio** (CV 5-fold, n=314):

```
ley     global  local pIns  ganancia  azar(control negativo)
pFe      0.352       0.496    +0.144        -0.022
pCu      0.690       0.770    +0.080        -0.024
pMo      0.345       0.341    -0.004        -0.016
pZn      0.360       0.310    -0.050        -0.036
```

El efecto es grande y aparece justo en Fe y Cu, los dos que no funcionan. Mo no
lo necesita: su señal es directa (canal propio, CV 72%).

**Pero el espectro no puede identificar el régimen:**

```
Mejor asidero de la ganga en el espectro:  |corr| = 0.283  (SumI vs pIns)
Predecir insolubles desde intensidades:     R2 = 0.088
Ruteo por insolubles ESTIMADOS:             0.311  (peor que global 0.352)
Acuerdo tercil real vs estimado:            41.4%  (azar = 33%)
```

Hay 5 canales: Fe, Cu, Zn, Mo, dispersión. Ninguno mide silicatos. El régimen
está codificado en la ganga, y la ganga es invisible para el analizador. **Es un
límite físico, no de método.**

**Por qué los clusters actuales no sirven:**

1. Parpadean: racha mediana de 3 lecturas ≈ 1.1 h. Un régimen dura días.
2. No separan la química: dispersión entre/dentro = 0.21 en pIns, 0.05 en pSol.
3. Los coeficientes por cluster difieren (pFe, `n1fe_ortho`: 0.70 / 0.72 / 1.70)
   pero no replican: el modelo con interacción da −0.007 fuera de muestra. Con
   ~105 muestras por cluster esa dispersión es ruido de estimación.

**Descriptores temporales de régimen tampoco aportan** (medianas móviles de las
fracciones, estrictamente pasadas):

```
base 0.514 | +móvil 3d 0.505 | +móvil 7d 0.505 | +móvil 21d 0.464 | +interacción -0.130
```

Autocorrelación diaria de las fracciones: lag1d 0.37–0.75, lag7d 0.02–0.14,
lag30d ≈ 0. No hay regímenes de semanas visibles en el espectro.

### 2.7 Detección de ruptura en lugar de identificación de régimen

Los operadores del courier generan ecuación nueva cada mes, corrigen offset a
diario contra el lab de 12 h, y generan modelo nuevo cuando cambia el mineral.
Eso es **detección de cambio**, no clasificación. CUSUM sobre los residuos:

```
estrategia                                R2    MAE   corr   pend  n_cal  n_gat
calendario 30 d                        0.233  0.918  0.500  0.314      5      0
calendario 15 d (actual)               0.244  0.914  0.508  0.317      9      0
calendario 7 d                         0.238  0.919  0.504  0.317     19      0
calendario 3 d                         0.252  0.909  0.515  0.326     41      0
CUSUM h=5 + tope 30 d                  0.263  0.895  0.521  0.320      4      1
CUSUM h=3 + tope 30 d                  0.264  0.900  0.523  0.323      3      4
CUSUM h=3 + calendario 15 d            0.261  0.901  0.521  0.324      8      3
CUSUM h=2 + calendario 7 d             0.256  0.905  0.518  0.326     18      4
```

**Mejor desempeño con menos recalibraciones** (7 vs 9). El CUSUM detectó 4 cambios
en 4 meses ≈ una cadencia mensual, que coincide con la práctica de los operadores.

**Honestidad estadística**: +0.015 en correlación con n=228 está dentro del ruido
para una comparación aislada (error estándar ≈ 0.05). Lo que convence es que las
cuatro variantes con CUSUM superan a las cuatro por calendario sin excepción. La
mejora sólida es de **eficiencia operativa**, no de exactitud.

**NO implementado todavía.** Ver hilos abiertos.

### 2.8 Redes neuronales y PINN: descartados con medición

Split temporal 70/30, n_train=538, n_test=231:

```
===== CU =====                              R2    MAE   corr  slope
crudas + Ridge                           0.139  1.333  0.517  0.206
razon I/Isc (fisica XRF)                -0.440  1.785  0.414  0.203
Lachance-Traill (correccion de matriz)  -0.794  2.026  0.376  0.256
MLP(16,)                                -2.463  2.834  0.209  0.231
MLP(64,32)                              -2.923  3.086  0.166  0.165
MLP(128,64,32)                          -3.050  3.136  0.232  0.254

===== FE =====
crudas + Ridge                           0.161  0.929  0.450  0.157
estequiometrico (Cu + Fe_exc)            0.075  0.990  0.454  0.142
Lachance-Traill                         -2.357  1.830  0.062  0.073
MLP(64,32)                              -5.319  2.532 -0.072 -0.110
```

El orden es monótono en simplicidad. Un PINN además es un error de categoría: no
hay EDP que imponer, es una regresión estática. Lachance-Traill falla porque sus
coeficientes asumen instrumento estable y el tuyo deriva 22%/año.

### 2.9 Estequiometría calcopirita/pirita

```
Fe total       : 27.91 +- 1.41
Fe calcopirita : 22.29 +- 1.79    (0.879 * Cu, por CuFeS2)
Fe excedente   :  5.62 +- 2.39    (pirita/otros)
corr(Cu, Fe)     = -0.106
corr(Cu, Fe_exc) = -0.811
```

Interpretación mineralógica válida: la dilución por pirita gobierna la ley de
cobre. **Pero NO explica el mal desempeño del Fe** — si lo explicara, la
calibración de fábrica tampoco podría. Reparametrizar (predecir Cu y Fe_exc,
reconstruir Fe) da 0.075 contra 0.161 del directo: no ayuda.

### 2.10 El sólido de laboratorio no ayuda al hierro

`pSol` es valor de laboratorio, no existe en línea. Sólo se usaría estimándolo
desde las mismas intensidades, y ese estimado es función lineal de las features
que el modelo ya tiene. Backtest:

```
variante de uso de pSol                       R2    MAE   corr   pend
ninguna (ruta B actual)                    0.244  0.914  0.508  0.317
pSol_est como feature adicional            0.243  0.914  0.507  0.316
I_i / pSol_est (correccion de masa)       -0.196  1.063  0.321  0.278
log(pSol_est) como feature                 0.183  0.931  0.482  0.339
```

Además el modelo auxiliar extrapola mal: produjo `pSol_est` de hasta −13.4%.

Calidad del dato: `pSol` tiene un valor de 295%, un 0, y 4 bajo 5%. Limpiando a
`5 < pSol <= 100` quedan 309 de 314.

---

## 3. Correcciones al razonamiento durante la sesión

Tres veces el usuario corrigió una premisa y el diagnóstico cambió. Quedan
registradas para no repetirlas.

| # | Lo que se afirmó | Corrección | Impacto |
|---|---|---|---|
| 1 | "`works4cdp_assay` no es laboratorio; corr 0.436 con el compósito es techo de etiqueta" | Ambas son laboratorio. La discrepancia es grab vs compósito de 12 h: variabilidad de proceso, no error | El consejo "reentrenar contra otra etiqueta" era erróneo |
| 2 | "Fe es irresoluble, la fábrica también falla" | Se miró el R² (−0.485) sin separar sesgo de correlación. En correlación la fábrica saca 0.470 y el modelo 0.278 | Fe pasó de "límite físico" a "el mayor margen disponible" |
| 3 | "La arquitectura local no aporta, la hipótesis de la tesis no se sostiene" | Correcto sobre la implementación, falso sobre la hipótesis: con insolubles reales da +0.144 | La hipótesis se sostiene; falla la variable de ruteo |

**Lección transversal**: separar siempre sesgo de correlación antes de concluir.
Un R² negativo puede ser puro sesgo y esconder una correlación utilizable.

---

## 4. Los cuatro cambios implementados

Ver `notebooks/10_recalibracion_composito.py` y `docs/flujo_pipeline.md`.

1. **Entrenar Fe con el compósito de 12 h** (760 ventanas) en vez de las 314 puntuales.
2. **Ventana móvil de 270 días**, reentreno cada 15 días.
3. **Corrección de sesgo en línea**, últimos 20 compósitos, separada por turno
   (el sesgo de Fe difiere 0.211 entre día y noche).
4. **Detector de anomalías sobre fracciones de cierre** en vez de intensidades
   absolutas: rechazo de 31% → 2.7%.

**Solo se recalibró Fe.** Medido:

```
ley   antes   recalibrado   fábrica   techo   decisión
pFe   0.278       0.508       0.470   0.743   RECALIBRAR
pCu   0.505       0.389       0.517   0.634   NO -- empeora
pMo   0.958       0.951       0.948   0.961   NO -- ya en el techo
```

Cu empeora porque su modelo original se calibra con muestras puntuales
emparejadas al instante exacto de una lectura; para cobre esa señal es más limpia
que el promedio de una ventana de 12 h.

Barridos que fijaron los parámetros (backtest walk-forward sobre pFe):

- ventana: 120 d → 0.465 | 180 d → 0.489 | **270 d → 0.508**
- reentreno: 30 d → 0.500 | **15 d → 0.508**
- features: sin `fe_cu` → **0.508** | con `fe_cu` → 0.488 | `n6sc` indiferente

---

## 5. Hilos abiertos

### 5.1 Integrar el CUSUM como quinto cambio
Reemplazar el reentreno por calendario con gatillo por deriva de residuos.
Medido: 0.508 → 0.523 con 7 reajustes en vez de 9. Código de prueba en la sesión,
**no integrado al pipeline**.

### 5.2 Arreglar el filtro de sensor congelado
Rechaza 6.6% de lecturas válidas. Los datos de PI vienen forward-filled en grilla
de 15 min, así que la repetición es por diseño del tag, no falla del sensor.

### 5.3 Ortogonalizar contra `SumI` en vez de `n6sc`
`SumI` sí correlaciona con el sólido (0.659) y `n6sc` no (−0.065). La ruta A (Cu,
Mo) sí usa ortogonalización. **No probado.**

### 5.4 Ruteo por variables de planta (solo producción)
Restricción del proyecto: **los datos de proceso son sensibles y no se pueden
usar en la tesis**; las intensidades sí se pueden anonimizar. Para producción sí
está permitido.

Con insolubles reales el ruteo da +0.144 en Fe. Las variables de planta que
deberían rastrear el régimen: potencia de molino y kWh/t, presión y energía
específica del HPGR, carga circulante y de zarandas, torque de espesador y dosis
de floculante, granulometría en línea.

Experimento propuesto: extraer 6–10 tags de PI desde 2025-07, alinear a las
ventanas de 12 h, medir (a) cuánto predicen el insoluble de laboratorio y (b) si
rutear por ellas reproduce el +0.144 en el backtest contra el compósito. Si (a)
da |corr| > 0.6, hay camino para llevar el Fe hacia el techo de 0.743.

### 5.5 Validar pZn o sacarlo de resultados
No tiene ninguna validación contra el compósito, que no trae Zn. Contra las 314
puntuales con split temporal: PLS(2) R²=0.349, RidgeCV R²=0.396.

### 5.6 Unificar el conjunto de evaluación
Cu aparece como 0.505 (colocación triple, 184 ventanas) y 0.484 (evaluación
final, 228 ventanas). Es diferencia de muestra, no de modelo. Para la tesis hay
que citar uno solo — se sugiere el de 228.

---

## 6. Restricciones y límites de alcance

1. **Datos de proceso prohibidos en la tesis.** Sensibles. Las intensidades se
   pueden anonimizar, las de proceso no. Sí permitidos para producción.
2. **Todo descansa en 13 meses.** Antes de 2025Q3 hay ~25 lecturas de intensidad
   por trimestre contra miles después. El dataset alineado arranca en 2025-07 y
   la evaluación cubre abr–ago 2026.
3. **La ruta B requiere mantenimiento cada 15 días.** Si nadie lo sostiene en
   planta, el 0.508 se degrada.
4. **`assay_lab_AIQ_pi.csv` NO es laboratorio.** 238 registros/día,
   autocorrelación lag-1 de 0.907. Es la salida del analizador con su calibración
   de fábrica. Úsese como línea base competidora, nunca como referencia.

---

## 7. Encuadre sugerido para la tesis

De "sistema de estimación multielemento" a **"caracterización del límite de
estimación de un analizador en línea, con un caso de éxito en molibdeno"**.

- **Mo** es el resultado principal: 99.6% de la señal disponible capturada,
  supera a la calibración de fábrica.
- **Fe** es el resultado metodológico: recalibración + ventana móvil + corrección
  de sesgo recuperan casi el doble de señal y superan al fabricante.
- **Cu** va como límite documentado, con techo 0.634 y explicación física.
- **Zn** sale de resultados o va explícitamente sin validación independiente.
- **Etapas 2–4** se reportan como hipótesis arquitectónica evaluada que **no** se
  sostuvo con ruteo espectral, con el +0.001 como evidencia — y con el +0.144 por
  insolubles como demostración de que el efecto existe pero el instrumento no
  puede explotarlo.

Aportes defendibles: cuantificación del techo por colocación triple, superar la
calibración comercial en los tres elementos en MAE, diagnóstico de deriva con
detector inmune, formalización de la práctica manual de los operadores (ecuación
mensual + offset diario), y un conjunto de resultados negativos rigurosos.
