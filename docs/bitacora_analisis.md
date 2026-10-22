# Bitácora de análisis — diagnóstico y recalibración del pipeline

Registro del razonamiento, las mediciones y las decisiones tomadas durante la
sesión de diagnóstico. Sirve para retomar el trabajo en otra máquina sin repetir
los experimentos. **Todos los números de este documento están medidos**, no
estimados; cada uno indica el protocolo con que se obtuvo.

Última actualización: 2026-10-01

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

**Si falta `assay_lab_courier_pi.csv` pero existe `data/raw/LABCOMPOSITO.csv`**
(export directo de PI con los tags sin renombrar, vía
`scripts/extraer_labcomposito.py`): regenerarlo con el mapeo `TAGS_COMPOSITO`
de `config.py`, no hay que volver a extraer de PI —

```python
import pandas as pd
from src.pipeline.config import TAGS_COMPOSITO, DATA_RAW
lab = pd.read_csv(DATA_RAW / "LABCOMPOSITO.csv").rename(columns={"t": "idx", **TAGS_COMPOSITO})
lab = lab.set_index("idx"); lab.index.name = None
lab.to_csv(DATA_RAW / "assay_lab_courier_pi.csv")
```

Verificado en sesión (2026-09-28): reproduce el backtest oficial exacto
(corr pFe = 0.508). Ver §8.5.

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

> **Corregido el 2026-10-01 — la conclusión de esta última frase estaba mal
> leída, no los números.** Los números se reprodujeron (§13): lag7d 0.04–0.17
> sobre las fracciones de grupo. Pero una autocorrelación nula a 7 días es
> **exactamente lo que produce un régimen de 3-5 días**, no la ausencia de
> régimen; y el espectro habla de **periodicidad** (no hay ciclo semanal), no de
> **persistencia**. Sí hay régimen de mineral, y cambia cada 1-2 semanas a
> resolución semanal: ver §13.

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

### 2.11 Los insolubles SÍ están en el PI (hallazgo del 2026-09-26)

Al extraer los tags del compósito directamente del PI se verificó el descriptor
real de cada uno, que hasta entonces se había **deducido, no confirmado**:

```
7100AIP101MAN = %Fe Promedio Concentrado colectivo  Courier Cobre
7100AIP102MAN = %Cu Promedio Concentrado colectivo  Courier Cobre
7100AIP103MAN = %Ins Concentrado colectivo  Courier Cobre      <-- INSOLUBLES
7100AIP104MAN = %Moly Promedio Concentrado colectivo  Courier Cobre
```

Dos consecuencias:

**1. El mapeo documental de `config.py` estaba mal** — tenía Fe y Cu invertidos y
suponía un tag de Zn inexistente. Error solo documental: ninguna etapa lo usaba.
Se verificó que las columnas de `assay_lab_courier_pi.csv` están bien nombradas,
idénticas al PI hasta 1e-6. **Todos los resultados anteriores se mantienen.**

**2. Los insolubles están disponibles para los 2594 compósitos**, no solo para
las 314 muestras puntuales. Media 9.40%, sd 2.75, rango 2.02–23.68, desde
2022-12-30, misma cadencia de 12 h.

Esto cambia el estado del hilo 5.4. En la sección 2.6 se concluyó que el ruteo
por insolubles daba +0.144 en pFe y +0.080 en pCu pero **no era desplegable**,
porque el insoluble solo existía en las 314 puntuales y no se puede predecir
desde el espectro (R² = 0.088). Eso era cierto para la fuente que se conocía
entonces. Con el tag del PI:

- Es **dato de laboratorio**, no de proceso -> permitido en la tesis.
- Llega en la **misma cadencia de 12 h** que la corrección de sesgo y el
  reentreno, así que encaja en el ciclo operativo ya implementado.
- En inferencia no se conoce el insoluble del turno en curso, pero sí el del
  turno anterior. Los regímenes duran días, así que el último valor conocido es
  un indicador razonable del régimen actual — y es exactamente la información
  que tiene el operador cuando decide generar una ecuación nueva.

Correlaciones del insoluble con las leyes del compósito (n=2590):
`corr(Ins, Cu) = -0.565`, `corr(Ins, Fe) = -0.476`. Consistente con la dilución
por ganga bajando la ley.

**PENDIENTE DE MEDIR**: si rutear por el insoluble del turno anterior reproduce
el +0.144 en el backtest walk-forward contra el compósito. Las ganancias de la
sección 2.6 están medidas en CV sobre las 314 puntuales, que ya se comprobó que
es optimista.

Datos en `data/raw/LABCOMPOSITO.csv`, extraídos con
`scripts/extraer_labcomposito.py`.


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

### 5.4 Ruteo por régimen de mineral — REABIERTO con los insolubles del PI

**Prioridad alta.** Ver sección 2.11: el insoluble está en el tag 7100AIP103MAN
para los 2594 compósitos, es dato de laboratorio (permitido en tesis) y llega
cada 12 h. Medir si rutear por el insoluble del turno anterior reproduce el
+0.144 de pFe en el backtest contra el compósito.

**Actualización 2026-09-28 — CERRADO, no se sostiene.** Se midió, con el
backtest walk-forward real y contra un control negativo (mismo tamaño de
grupos, sin información real): el control iguala o supera al insoluble real en
Fe, Cu y Zn; solo en Mo el insoluble real queda 0.03 por delante, dentro del
ruido que ya muestra ese mismo control entre sus propias celdas (0.15 a 0.97).
El +0.144 de la sección 2.6 era CV 5-fold con el insoluble **simultáneo**
(no disponible en producción); con el insoluble del turno anterior y
validación honesta, el efecto no se distingue de repartir clusters al azar.
Detalle completo, metodología y las otras tres pruebas relacionadas en §8.

### 5.4b Ruteo por variables de planta (solo producción)
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

**Actualización 2026-09-28.** La premisa (+0.144 reproducible) queda retirada
por §5.4. Este experimento pierde su justificación tal como estaba planteado
--- si se retoma, hay que remedir el punto (b) contra un control aleatorio
antes de invertir en extraer los 6-10 tags de planta.

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
  sostuvo con ruteo espectral, con el +0.001 como evidencia. El insoluble real
  del turno anterior **tampoco** sostiene el ruteo bajo validación honesta
  (§5.4, §8): el efecto de régimen sigue siendo real en principio (§2.6), pero
  ninguna variable disponible hoy —ni el espectro, ni el insoluble rezagado, ni
  la cabeza del circuito— permite explotarlo en la práctica.

Aportes defendibles: cuantificación del techo por colocación triple, superar la
calibración comercial en los tres elementos en MAE, diagnóstico de deriva con
detector inmune, formalización de la práctica manual de los operadores (ecuación
mensual + offset diario), y un conjunto de resultados negativos rigurosos.

---

## 8. Sesión 2026-09-28 — cierre del hilo del insoluble y del régimen de mineral

Cuatro pruebas distintas de usar el insoluble (o una fuente similar) para
mejorar clustering/ruteo. Las cuatro veces, un control negativo (mismo tamaño
de grupo, sin información real) igualó o superó a la señal real. Metodología y
cifras completas de cada una, en el orden en que se corrieron.

### 8.1 Bug de datos encontrado y corregido: columnas `_ortho` mal etiquetadas

`data/processed/intensidad_cobre_24_completo_filtrado.csv` (y `_completo.csv`)
traían columnas `n1fe_ortho..n4mo_ortho` que **no son el ortogonalizado crudo**
(salida de la etapa 2) sino el **ya escalado** (salida de la etapa 3) —mismo
nombre en dos archivos, dos significados. Confirmado con un caso concreto
(timestamp 2022-12-05 15:38:30): `n1fe_ortho = 8378.12` en
`intensidad_cobre_24_orthogonalized.csv` (crudo) vs `n1fe_ortho = 3.16` en el
archivo supervisado (escalado).

**Impacto real: ninguno en el pipeline de producción.**
`src/pipeline/modelos.py:_recomputar_features` nunca lee esas columnas —
siempre recalcula ortho y cluster desde las intensidades crudas con los
artefactos congelados, que es como debe ser para que entrenamiento e
inferencia sean idénticos. El riesgo era solo para quien reutilizara esas
columnas a mano asumiendo que eran crudas (aplicar power+scaler otra vez sobre
un valor ya escalado colapsa el GMM a un solo cluster — así se detectó).

**Corregido:** se eliminaron esas 4 columnas de ambos CSV y se actualizó
`src/pipeline/dataset_supervisado.py` para que no vuelva a arrastrarlas. Las
columnas `cluster_kmeans`/`cluster_gmm` sí se conservan (son etiquetas, no
features, y sí las usa el pipeline).

### 8.2 Prueba 1 — Ruteo de pFe por insoluble del turno anterior (walk-forward)

Rediseño honesto de la sección 2.6: `ins_lag` = insoluble del compósito
**anterior** (nunca el simultáneo, que no existe en producción), terciles
calculados **solo con la ventana móvil vigente** en cada reentreno (270 d,
igual que la recalibración de Fe), backtest walk-forward sobre las mismas 228
ventanas fuera de muestra.

```
                  R2 global   R2 ruteo   ganancia
ruteo real (ins_lag)  0.160      0.242     +0.082
control (barajado)    0.160      0.220     +0.061
```

Ganancia real vs. control: **+0.021** — dentro del ruido. El +0.144 de la
sección 2.6 (CV 5-fold, insoluble simultáneo) no sobrevive ni al cambio de
validación ni al cambio a insoluble rezagado; no se separó cuál de los dos
pesa más. Código: `src/pipeline/ruteo_insoluble.py`,
`notebooks/12_experimento_ruteo_insoluble.py`.

### 8.3 Prueba 2 — Cabeza (rebose de hidrociclones) como feature de clustering

Se exploró una segunda fuente de régimen: el rebose de hidrociclones (cabeza
del circuito, antes de flotación), streams `Rebose Hidrociclones L1/L2`
(`works4cdp_assay.sample_id` 17 y 18, mismo `equipment_id=5` que el
Concentrado Colectivo).

- **L1**: 384 filas históricas pero solo **24 en los últimos 12 meses**
  (15 con `pIns`) — casi sin dato reciente.
- **L2**: 406 filas, 85/año recientes, pero `corr(n2cu_L2, pCu_lab) = 0.04`
  crudo — el usuario confirmó que hubo cambios de calibración en el histórico
  de intensidades de esta línea, lo que explica la correlación cruda pobre.

Con L2 alineado al stream de 15 min por último valor conocido (cadencia
mediana 72 h → cada lectura de cabeza se repite ~86 veces seguidas): el BIC se
desploma con la cabeza real, **pero colapsa exactamente igual con la cabeza
barajada** (mismo orden de magnitud) — es el artefacto de bloques repetidos,
no información real.

Repitiendo a la granularidad correcta (una fila por época de cabeza, ~266
filas en vez de 23,058): el traslape real de alta frecuencia es mínimo (solo
58 de 267 épocas con ≥5 lecturas de courier, porque el courier de alta
frecuencia solo existe desde 2025-07 y la cabeza tiene historia desde 2021).
Con n=58 el resultado es ruido en ambas direcciones (real pierde en silhouette
GMM contra el control, gana en BIC) — **inconcluso, no descartado**: falta
traslape de alta frecuencia, no evidencia de que no sirva. Ver §8.4 sobre por
qué no hay una vía de mayor frecuencia disponible para esta variable en
particular (si la hay para la cabeza, no se investigó).

Código: `notebooks/13_experimento_cluster_cabeza.py`,
`notebooks/14_experimento_cluster_cabeza_epocas.py`.

### 8.4 Confirmación: el courier NO mide insolubles

Se investigó si el analizador mismo genera una intensidad de insoluble
(`a7a7`/`a6sol` en `works4cdp_assay`) que sirviera de proxy de alta frecuencia,
evitando la cadencia de 12 h del compósito de laboratorio.

- El canal crudo análogo a `n1fe..n4mo` para insoluble es `n7ech7` — está
  **0/548 no nulo** para `sample_id=24` (Concentrado Colectivo,
  `equipment_id=5`). Sí existe en otros equipos del mismo analizador
  multiplexado (`equipment_id=1`: ~70-80% de llenado), pero no en el stream
  que le interesa a este proyecto.
- `a7a7`/`a6sol` sí tienen valores reales (no son el centinela `-1`) para
  `sample_id=24`, pero **confirmado directamente con el equipo/operador**: son
  una estimación que el operador calcula a partir de las demás leyes (Fe, Cu,
  Zn, Mo), no una lectura del sensor. Medido: `corr(a7a7, pIns_lab) = 0.116` —
  consistente con que es una fórmula, no una medición, y con el R²=0.088 de la
  sección 2.11 (predecir insoluble desde el espectro).

**Conclusión: no hay atajo de alta frecuencia para el insoluble en este
stream.** La única fuente legítima sigue siendo el compósito de laboratorio vía
PI (12 h) o las muestras puntuales (`pIns` en la BD).

### 8.5 Prueba 3 — Insoluble en el clustering (pipeline 0.5→1→2→3→4)

Diseño acordado: `ins_t` (último insoluble conocido, merge_asof hacia atrás)
entra en la etapa 3 (Yeo-Johnson + StandardScaler) junto a las 4 features
ortogonalizadas de siempre; etapas 1-2 sin cambios. Evaluado con
`eta² = SS_entre_clusters / SS_total` contra las leyes reales de las 314
muestras puntuales (recalculando el ortogonalizado crudo por timestamp desde
el stream — ver §8.1, el bug que esto destapó).

```
ley    oficial(prod)  base(refit)  +ins_t real  +ins_t control
pFe        0.194         0.215        0.176          0.212
pCu        0.140         0.135        0.056          0.135
pMo        0.147         0.140        0.117          0.137
pZn        0.065         0.073        0.095          0.074
```

El insoluble real separa **peor** Fe, Cu y Mo que sin él, y el control
(insoluble barajado en el entrenamiento) rinde igual que sin insoluble en las
tres. Solo Zn mejora un poco, con eta² bajo en todos los casos (más
compatible con ruido). Código:
`notebooks/15_pipeline_insoluble_clustering.py`.

### 8.6 Prueba 4 — Insoluble como variable de ruteo en la etapa 5 (modelos locales)

Última prueba, con la métrica que de verdad importa: R² de los modelos locales
contra ley real, mismo `KFold(5, shuffle=True)` que ya usa `modelos.py`, única
diferencia el cluster usado para rutear.

```
ley   R2 local (oficial)  R2 local (+ins_t real)  R2 local (control barajado)
pFe         0.161                 0.218                    0.305
pCu         0.392                 0.500                    0.604
pMo         0.620                 0.659                    0.625
pZn         0.248                 0.228                    0.334
```

El control aleatorio **gana en 3 de 4 leyes**, con margen (hasta −0.10 en Cu).
Solo en Mo el insoluble real gana por +0.033 — dentro del ruido que ya muestra
el propio control entre sus celdas individuales (R² de 0.15 a 0.97 en la misma
corrida). Veredicto: **no rutear por insoluble.** Figuras:
`reports/fig_insoluble_vs_control.png`, `reports/fig_tendencia_insoluble.png`.
Código: `notebooks/16_modelos_locales_con_insoluble.py`,
`notebooks/17_control_ruteo_insoluble.py`, `notebooks/18_*`, `notebooks/19_*`.

### 8.7 Reconstrucción de `assay_lab_courier_pi.csv` y verificación cruzada

Este archivo (esperado por `cargar_composito()`/celda 10) no existía en este
clon del repo — solo `data/raw/LABCOMPOSITO.csv` (export crudo de PI, tags sin
renombrar). Se regeneró aplicando el mapeo `TAGS_COMPOSITO` ya definido en
`config.py` (ver receta en §0). Al correr `notebooks/10_recalibracion_composito.py`
con el archivo regenerado, el backtest reprodujo **exacto** lo ya documentado
(`corr pFe = 0.508`, fábrica 0.470) — verificación cruzada de que
`LABCOMPOSITO.csv` y el `assay_lab_courier_pi.csv` original son la misma
fuente, solo con formato distinto.

### 8.8 Figura de validación filtrada desde 2026-05-01 ("la mejor estrategia")

Con el backtest reconstruido, se regeneró la evaluación unificada de la celda
11 (Fe recalibrado + Cu/Mo modelo original, contra el mismo compósito)
filtrada a partir del 2026-05-01 (216 de las 228 ventanas totales, que ya
cubrían casi por completo ese rango):

```
elemento   n    R2    MAE    corr   pendiente
Fe        216  0.254  0.921  0.516    0.324
Cu        216  0.071  1.417  0.470    0.251
Mo        216  0.905  0.162  0.953    0.882
```

Consistente con las cifras de todo el período abr-ago. Código:
`notebooks/20_figura_validacion_desde_mayo.py`, figura en
`reports/fig_validacion_desde_mayo.png`.

---

## 9. Sesión 2026-09-29 — corrector Kalman, recuperación de filas, y dos hilos más cerrados

### 9.0 Recomendación — la mejor opción para el mejor resultado

De todo lo medido esta sesión (aquí + §8), **una sola cosa vale la pena
adoptar en producción**: el corrector Kalman de sesgo para **pFe**
(§9.2) — mejora corr, R² y MAE a la vez, mismo backtest, sin tocar el
modelo. El resto son negativos ya sea confirmados (insoluble, cabeza,
orden escalado/ortogonalización) o resultados mixtos que requieren una
decisión editorial, no una adopción automática (Kalman en Cu, §9.3). El
filtro de `-9999` recuperable (§9.1) es una corrección de robustez, no
cambia ningún número hoy. `pSol` de alta frecuencia (§9.5) sigue
bloqueado por falta de instrumentación/ingesta.

```
cambio                              estado          accion recomendada
------------------------------------------------------------------------
Kalman en pFe (bias, q=0.1)         MEJORA (3/3)    ADOPTAR
Kalman en pCu (bias, q=0.0001)      MIXTO           decidir por criterio
                                                      (R2/MAE si o corr no)
filtro -9999 recuperable            robustez        ya aplicado, sin riesgo
orden escalado/ortogonalizacion     NO AYUDA         dejar como esta
insoluble (4 pruebas, sesion previa) NO AYUDA        cerrado (S8)
cabeza/rebose hidrociclones          INCONCLUSO      pendiente de mas dato
pSol alta frecuencia                 NO EXISTE        pendiente de ingesta
```

### 9.1 Filtro de limpieza: filas con error parcial ya no se borran

`_filtrar_invalidos()` (`src/pipeline/limpieza.py`) borraba la fila
completa si **cualquiera** de los 5 canales traía el código de error
`-9999`, perdiendo los otros 4 canales válidos por uno malo. Ahora solo
se borra si **los 5** están en error; si es parcial, el canal puntual
pasa a `NaN` explícito y la fila se conserva. Se corrigió también
`limpiar()` para que esas filas recuperadas (no evaluables por el
IsolationForest, por el `NaN`) no se pierdan igual dos pasos después --
antes se descartaban en silencio sin dejar rastro. De paso, un bug
latente en `features_cierre()` (`features.py`): la suma de cierre usaba
`skipna=True` por defecto, así que un metal en `NaN` no anulaba la
fracción de los otros 3 (las calculaba con un denominador entendido de
menos) -- corregido con `min_count=len(METALES)`.

**Con los datos actuales, esto no cambia ningún número**: las 4 filas
con `-9999` de hoy tienen los 5 canales en error, no parcial. Verificado
con 4 casos sintéticos (parcial en `n6sc`, parcial en un metal, total,
y el caso real sin cambios) -- los 4 se comportan como se espera.
Corrección preventiva/estructural, no retroactiva.

### 9.2 Corrector Kalman de sesgo para pFe -- RECOMENDADO ADOPTAR

`CorrectorSesgo` (el "cambio 3" de la recalibración de Fe) promedia los
últimos 20 residuos con ventana fija -- viola la regla del proyecto de
usar EWMA, nunca media móvil simple. Se implementó `CorrectorKalman`
(`src/pipeline/corrector_kalman.py`): sesgo como caminata aleatoria
escalar por `(ley, turno)`, con el ruido de proceso escalado por tiempo
transcurrido (maneja el muestreo irregular, que una ventana de N no
puede). Misma interfaz que `CorrectorSesgo`, inyectable en
`calibracion_composito.backtest(corrector=...)` sin tocar esa función
(comportamiento por defecto sin cambios si no se pasa nada).

Backtest walk-forward, mismos 228 puntos que ya valida el resto del
pipeline:

```
                          corr      R2     MAE
media movil (actual)     0.508   0.244   0.914
kalman q=0.1 (nuevo)     0.576   0.316   0.863
diferencia              +0.068  +0.072  -0.051
```

Barrido de `q_por_dia` (único hiperparámetro libre, r=1 fijo): sube
monótono hasta saturar en 0.1-0.2 y **cae de nuevo** en q>=0.5 pese a
que el corr seguía subiendo un poco ahí -- se eligió por R2, no por la
métrica que más convenía. Las tres métricas mejoran a la vez, a
diferencia de todo lo demás probado esta sesión. Figuras:
`reports/fig_tendencia_kalman_vs_lab.png` (Fe solo, media móvil vs
Kalman vs lab) y `reports/fig_validacion_kalman.png` (los 3 elementos,
mismo formato que la celda 11, con Fe corregido por Kalman: corr Fe
0.576, Cu 0.484 y Mo 0.953 sin cambios). Código:
`notebooks/21_backtest_corrector_kalman.py`,
`notebooks/22_figura_validacion_kalman.py`.

**No implementado en el bundle de producción todavía** --
`models/calibracion_composito.joblib`/`corrector_sesgo.joblib` siguen
con `CorrectorSesgo`. Adoptar significa cambiar
`notebooks/10_recalibracion_composito.py` para usar `CorrectorKalman`
en el paso 4 y volver a correr el ciclo completo.

### 9.3 Corrector Kalman de sesgo para pCu -- resultado mixto, no automático

Cu **nunca tuvo** un corrector de sesgo (el "cambio 3" solo se aplicó a
Fe), pese a que la bitácora ya medía que el sesgo de Cu también
deambula por trimestre incluso en la calibración de fábrica (§2.3:
`sesgo_cu` 0.462 -> 1.109 -> 0.055 -> 0.018). Se probó el mismo
mecanismo, sin tocar el modelo de Cu:

```
                          corr      R2     MAE
sin corrector (actual)   0.484   0.086   1.390
kalman q=0.0001 (nuevo)  0.463   0.207   1.283
diferencia              -0.021  +0.120  -0.107
```

`q` óptimo confirmado en el límite q->0 (no es borde de grilla: 0.0001,
0.0005 y 0.0 dan el mismo resultado, 0.207/0.463/1.281-1.283). R² y MAE
mejoran sustancial -- el modelo de Cu tenía un sesgo negativo
persistente, visible en el panel de error de
`reports/fig_tendencia_kalman_cu_vs_lab.png`, y corregirlo ayuda mucho
a esas dos métricas, que son muy sensibles a sesgo sistemático. Pero
corr **baja** -- el corrector arregla el desplazamiento, no la forma de
la relación (que es lo que corr mide), y reaccionar a la deriva
reciente mete algo de varianza. Interpretación: Cu no falla por un
offset simple (como sí parece ser el caso de Fe), falla por ruido de
medición en el canal -- corregir el offset no resuelve el problema de
fondo. **Decisión pendiente de criterio editorial**: si lo que importa
para la tesis es exactitud absoluta (MAE/R²), adoptarlo; si es el techo
de colocación triple (corr), no. Código:
`notebooks/23_backtest_corrector_kalman_cu.py`.

### 9.4 ¿Escalar antes o después de ortogonalizar? -- no ayuda, y van 6 controles negativos seguidos

Solo podía importar para clustering (para regresión, §2.5 ya prueba que
es un no-op algebraico). Medido con GMM k=3, sobre las 314 muestras con
ley real, contra un control aleatorio del mismo tamaño de grupo:

```
ley     A (actual)   B (invertido)   control (azar, tam. B)
pFe        0.166         0.207              0.310
pCu        0.418         0.482              0.590
pMo        0.620         0.636              0.594
pZn        0.264         0.240              0.303
```

El control supera a B en 3 de 4 leyes -- la aparente mejora de invertir
el orden es ruido de partición, no señal del reordenamiento. **Dejar el
pipeline como está.**

**Hallazgo colateral, más importante que la pregunta original**: esta
es la **sexta** vez en la sesión (insoluble x4, cabeza, y ahora esto)
que un control aleatorio iguala o supera al cambio real bajo el
`KFold(shuffle=True)` de la etapa 5. Con el tamaño de muestra actual
(~307-314), casi cualquier repartición de cluster puede parecer una
mejora si no se contrasta contra el azar. Esto pone en duda no solo
cada cambio probado con esta validación, sino **el método de
evaluación de la etapa 5 en sí** -- que sigue siendo el mismo
`KFold(shuffle=True)` que ya se identificó como optimista en
`data2Tesis` (ADR de `decisiones.md`: "Validación temporal: de `KFold`
barajado a walk-forward con purga y embargo") y que **nunca se corrigió
en V2** para la etapa 5, solo para la recalibración de Fe contra el
compósito (celda 10). Queda como hilo abierto de mayor prioridad que
cualquiera de los específicos de esta sesión. Código:
`notebooks/24_orden_escalado_ortogonalizacion.py`,
`notebooks/25_impacto_orden_en_cu_mo.py`.

### 9.5 `pSol` de alta frecuencia -- no existe en ninguna fuente revisada

Se evaluó si había sólidos de laboratorio a alta frecuencia (para
ortogonalizar contra dilución real en vez de `n6sc`, que §2.4 ya probó
que no mide el sólido). Revisadas las dos fuentes:

- **Compósito de PI** (`LABCOMPOSITO.csv`, cada 12h): solo 4 tags --
  Fe, Cu, Ins, Mo (`7100AIP101-104MAN`). **No trae `pSol`.**
- **Base de datos** (`works4cdp_assay`): `pSol` existe, pero a la
  misma cadencia baja que el insoluble -- mediana de 67.8h entre
  lecturas, 366 muestras desde 2021-07 para `sample_id=24`
  (Concentrado Colectivo, el mejor de los 15 `sample_id` revisados).

**No hay sólidos de alta frecuencia en ningún sistema al que se tenga
acceso hoy.** Falta ingresar ese dato a la base de datos (instrumentación
u otra fuente) antes de poder medir si ayuda -- queda como hilo abierto,
bloqueado por adquisición, no por análisis.

---

## 10. Sesión 2026-09-30 — revisión etapa por etapa (0-4) y grupos de mineral

Pipeline reescrito desde la adquisición hasta el clustering, con datos hasta
2026-09-30. ADR correspondientes en `Proyects/docs/decisiones.md` (4 entradas
del 2026-09-30).

### 10.1 Etapa 0 — tres fuentes, un crudo cada una
`courier_bd.csv` (BD, 548 filas, 366 con ley), `courier_pi.csv` (PI
`recorded`, 413,834 eventos, desde 2025-07-13), `composito_pi.csv` (2,601
ensayos 12 h). Token PI por `PI_TOKEN`/`~/.pi_token`; pasarela 10.25.18.85:5173.
Los tags `_296290_ConcFinal_*` son el `sample_id=24`: 125/125 muestras calzan,
0 de ~2,700 de las otras 24 líneas.

### 10.2 Lo que es el PI, medido
- El courier da **una lectura del concentrado cada ~21 min** (mediana 1,251 s);
  el PI la re-archiva cada 100 s (`ExcMax=100` en los 5 tags) → ~14 eventos
  por lectura. No es sensor congelado.
- 35,925 eventos todo-0 (analizador detenido, 177 días) y 41 todo -9999.
- `n6sc` archivado 1-145 s aparte de los metales, concentrado en mar-jul 2026
  (pico 6,399 **filas crudas** en mayo = 442 lecturas; 1,310 en total).
  A veces `n1fe` llega 1 s antes que el resto.
  La 1a reconstruye: 25,429 lecturas hasta el 30-sep, 0 con metales incompletos.
  → **Rehecho el 2026-10-21: los 5 canales son simultáneos y la fila parcial
  no es una lectura. Ver §18; la 1a ahora da 25,427 lecturas.**

### 10.3 BD ↔ PI
Calce por valor de los 4 metales redondeados: **125/125** desde 2025-07-13.
(Decía que las 18 "perdidas" de mar-jul eran lecturas partidas por `n6sc`:
**ese número no es reproducible y el mecanismo no cierra** — ver §18.6.) Hora BD
~6 min detrás del inicio de la lectura en el PI (p95 28 min). **3 filas de
jun-2026 con la hora corrida exactamente 12 h** (AM/PM en la BD). 4 `instance`
duplicadas = misma lectura registrada con y sin leyes. `instance` no sirve de llave.

### 10.4 Leyes
Ley en 0 = no analizada (pZn 72 en 2023-06..2025-10: n3zn igual con y sin
pZn=0 → el Zn estaba ahí) → NaN. Ley sospechosa = |z| > 5 y su canal del
courier no la respalda (|z| ≤ 2 o signo opuesto): 4 filas — 2023-02-24
(pFe 4.16/pCu 0.43), 2025-04-18 (pMo 16.77), 2026-02-16 (pSol 295),
2026-05-29 (pFe 39.18). Los pZn/pMo altos reales traen su canal alto.

### 10.5 Etapa 1c — corte y anomalías
Corte train ≤ 2026-05-31, producción ≥ 2026-09-01. IsolationForest (fracciones
de cierre, contamination 2 %): train 2.0 %, test 9.2 % (junio 14.7 %,
Mo/Zn altos), producción 5.9 %. El compósito no decide si es mineral o
instrumento (corr fracción anómala vs Mo lab 0.06). `valida_ley`: 361/366.

### 10.6 Etapa 2 — por qué se eliminó la ortogonalización
Media por semestre de `n2cu_ortho`: +6,607 (2022H2) → −5,133 (2026H2), pCu
24.8 → 22.9. R² metal~n6sc 0.01-0.22. Log-cocientes vs química del lab:
Fe/Cu 0.73, Zn/Cu 0.86, Mo/Cu 0.95. GMM k=2 sobre intensidades absolutas
(lo de V1): silhouette 0.19, log(ΣI) igual en ambos grupos (11.20/11.18) →
**no se reproduce la separación por dilución de V1** con estos datos.

### 10.7 Dilución, medida
OLS `feature ~ química_lab + pSol` (n=343): lr_fe_cu t_pSol=+12.1
(p10→p90 = +0.91 sd), lr_mo_cu −10.5 (−0.37 sd), lr_zn_cu −1.3 (sin efecto).
Corrección con `dil = logSumI − EWMA(90 d, pasado)`: corr(dil, pSol)=0.50-0.55.
Aplicada solo a Fe/Cu (b=+0.104): 0.91 → 0.73 sd, corr química 0.75 → 0.80.
En Zn/Cu la corrección creaba un efecto (t=+3.3): no se aplica.

### 10.8 Etapa 4 — k y grupos

```
k   silhouette  ARI mitades     (con corrección de dilución)
2     0.290       0.958
3     0.230       0.661
4     0.249       0.856   <- elegido
5     0.227       0.541
```

| grupo | BD (n) | compósito (n) | pSol | train | sept |
|---|---|---|---|---|---|
| Cu | pCu 25.64 (117) | cu 25.77 (186) | 23.6 | 37 % | 18 % |
| Mo | pMo 2.41 (60) | mo 2.46 (207) | 21.5 | 30 % | 60 % |
| Zn | pZn 0.40 (61) | — | 20.0 | 19 % | 18 % |
| Fe | pFe 29.53, pCu 22.94 (123) | fe 28.77 (43) | 27.4 | 14 % | 5 % |

Septiembre por semana (% Mo del grupo / Mo compósito): 44/2.13, 95/2.71,
50/1.98, 53/1.73, 36/1.63 (hasta 30-sep 00:00: Cu 42→56 %). Racha mediana en
un grupo: 2 lecturas (~40 min), p90 21. Artifact:
https://claude.ai/artifact/LXzMFBmpFoooh5XwG88hfZ

### 10.9 Abierto
Etapa 5 (features y `valida_ley` nuevos, walk-forward), etapa 6,
`run_pipeline.py` y notebook 10 rotos por la nueva interfaz de 1-4.
Reconciliación con `tesis_ucspv2/` (etapa 2 y K=3 descritos allí ya no valen).

## 11. Sesión 2026-09-30 (2) — etapas 5, 6, 8 y 9: modelo vs laboratorio, sin y con Kalman

Cierra lo abierto en §10.9 salvo el notebook 10 y `run_pipeline.py`.
Commits `64ad2c3` y `8384dc7`. ADR en `../docs/decisiones.md` (2026-09-30).

### 11.1 Etapa 5 — walk-forward y control aleatorio
342 muestras de train con `valida_ley`; 5 folds de 57 (`split_purgado`, purga
y embargo 12 h: 0 filas purgadas, 1 intervalo < 12 h). N por grupo: Cu 112,
Fe 123, Mo 54, Zn 53 (pZn 83/101/45/41). Mo y Zn: global por N < 60. Cu y
Fe: ningún local supera al p95 de 200 particiones aleatorias. **Todo global.**
R² global fuera de fold (285 puntos): Fe −0.17, Cu 0.55, Mo 0.78, Zn 0.90
(215). Fe sobreestima +0.98 de forma sostenida: el Fe de la BD baja desde 2024.

### 11.2 Etapas 8-9 — contra el laboratorio
Compósito, fuera de muestra (jun-sep, n=209, sin anomalías); `q` elegido en train:

| ley | q | corr mod → kal | R² mod → kal | MAE mod → kal | sesgo mod → kal |
|---|---|---|---|---|---|
| Fe | 0.01 | 0.525 → 0.524 | −0.211 → 0.189 | 1.114 → 0.913 | −0.85 → −0.15 |
| Cu | 0.005 | 0.564 → 0.528 | 0.038 → 0.227 | 1.426 → 1.301 | −0.91 → −0.06 |
| Mo | 0.002 | 0.913 → 0.914 | 0.814 → 0.832 | 0.252 → 0.220 | −0.10 → +0.03 |

Con anomalías: casi igual (Mo corr 0.928). BD test (n=18): Fe R² 0.25 → 0.48,
Mo 0.63 → 0.64, **Cu 0.52 → 0.24**, Zn 0.83 (sin Kalman).

### 11.3 BD vs compósito
351 muestras de la BD contra el compósito de su ventana (07:30 cubre
01:30-13:30): BD − compósito Cu −1.23 (corr 0.48; por año −0.97/−1.20/−1.84/−0.74),
Fe −0.18 (0.61), Mo +0.07 (0.92). Explica el signo opuesto del sesgo del
modelo contra cada referencia y por qué el Kalman empeora Cu contra la BD.

### 11.4 Abierto
Referencia oficial para Cu (BD o compósito). Notebook 10 (recalibración de Fe
contra el compósito) y `run_pipeline.py` con la interfaz nueva. Inferencia en
línea. Sensibilidad del GMM por fold.

## 12. Sesión 2026-10-01 — origen operativo de las dos fuentes de laboratorio

Aclaración del usuario sobre de dónde sale físicamente cada ley, y la
verificación de lo que esa aclaración pone en duda. Todas las cifras de esta
sección las reproduce `scripts/verificar_fuentes_laboratorio.py` (solo lectura,
sobre las tablas limpias de la etapa 1a).

### 12.1 Las dos fuentes, y por qué hay dos rutas de calibración

**Ambas leyes salen del mismo courier**, pero de dos cortes físicos distintos:

| | BD (`sample_id=24`) | Compósito (PI) |
|---|---|---|
| Qué es | muestra **puntual** del concentrado colectivo, toma rutinaria | **compósito de 12 h** del concentrado colectivo / final |
| Dónde se corta | material que **sí pasó por los rayos X** | **antes del multiplexor**, antes de los rayos X |
| Intensidad propia | **sí** — existe la lectura del analizador que le corresponde | **no** — ninguna lectura le corresponde |
| Cadencia medida | **Lun/Mie/Vie**, 356/365 (97.5 %), intervalos modales 2 d y 3 d | **2/día**, marcas 07:30 y 19:30, los 7 días |
| Volumen | 365 muestras con ley, 2022-12-05 → 2026-08-26 | 2601 ensayos, ~95 % de los dos turnos, → 2026-09-29 |
| Leyes | Fe Cu Zn Mo Ins Sol | Fe Cu Mo Ins (**sin Zn**) |
| Hora de toma | 09-13 h: 297 · **15-19 h: 60** · otras: 8 | fija por turno |

Esto es la razón **física**, no metodológica, de los dos protocolos de
evaluación de §11.2, y conviene citarla así en la tesis:

- La muestra de la BD es el mismo material que midió el analizador, de modo que
  existe **una** lectura que le corresponde → el calce exacto por valor de los 4
  metales redondeados (§10.3, 125/125) no es un truco de emparejamiento: es
  identidad del material. Por eso la BD sirve para entrenar y evaluar por
  lectura.
- El compósito se corta **antes** de los rayos X, así que **no existe** lectura
  que le corresponda → la única alineación posible es promediar la estimación
  sobre su ventana. La ventana no es una elección entre varias: es la
  consecuencia del punto de muestreo.

Corolario ya conocido por otra vía: el compósito no reporta Zn, que es la razón
por la que el Zn no es auditable contra el laboratorio de referencia.

### 12.2 ¿La marca del compósito es el centro de su ventana o el cierre del turno?

La aclaración destapa una objeción que el pipeline no tenía respondida de forma
explícita. Si el compósito se **acumula a lo largo del turno** y se reporta al
cambio de turno, su soporte sería `[t-12h, t]` y la ventana **centrada**
`[t-6h, t+6h]` que usa todo el pipeline (`VENTANA_COMPOSITO_H`) estaría
desplazada 6 h — con una lectura cada ~21 min y régimen que cambia día a día,
6 h no es despreciable.

Medido **ley contra ley, sin modelo en medio** (cada muestra de la BD contra el
compósito que le asigna cada hipótesis; las 4 filas con ley sospechosa fuera):

```
hipótesis                 n        Fe               Cu               Mo
A centro [t-6h, t+6h]    351   -0.170 / 0.612   -1.243 / 0.476   +0.073 / 0.917
B cierre [t-12h, t]      350   -0.346 / 0.521   -1.447 / 0.365   +0.084 / 0.801
                                        (sesgo BD - compósito / correlación)
```

**A gana en las tres leyes.** El árbitro es **Mo**, la única ley en que ambos
laboratorios concuerdan (§11.3): 0.917 contra 0.801, sin ambigüedad. La marca
del compósito se comporta como el **centro** de la ventana que integra, no como
el cierre del turno.

Esto confirma por una vía independiente el barrido de lag documentado en
`src/pipeline/calibracion_composito.py` (lag −12 h a +12 h, óptimo en lag 0,
r_cu 0.345 contra 0.213 en lag −12 h): ahí se midió con intensidades y modelo,
aquí con laboratorio contra laboratorio. **La ventana centrada queda sostenida y
no hay que tocar nada** — se registra porque, dicha la cadencia operativa en voz
alta ("un compósito por turno, cada 12 h"), la ventana centrada *parece* mal
elegida, y es una objeción fácil de hacer en una sustentación.

### 12.3 El sesgo de Cu no es artefacto de alineación ni de hora de toma

Aporta a la decisión abierta de §11.4 (qué laboratorio manda para Cu):

```
BD tomada 09-13 h: n=285  sesgo Cu -1.121  corr 0.499
BD tomada 15-19 h: n= 58  sesgo Cu -1.813  corr 0.353
```

El sesgo persiste en ambas franjas y empeora por la tarde; y la asignación de
compósito que se usa es la correcta (§12.2). Entonces el −1.24 de Cu es
**diferencia sistemática entre los dos ensayos**, no un desalineamiento ni un
efecto diurno. Se separan dos cosas que conviene no confundir al redactar:

- la **diferencia de soporte temporal** (corte puntual vs integral de 12 h)
  explica la **correlación** baja — son dos cantidades distintas y ningún modelo
  puede acertar a ambas a la vez;
- el **nivel** (−1.24 puntos) no lo explica el soporte: promediar no corre la
  media. Queda como diferencia de ensayo, y sigue pendiente de metalurgia.

### 12.4 Consecuencia para `tesis_ucspv2/`

El Cap. 4 describe los datos como "BD + PI concatenados" (ya señalado en el ADR
del 2026-09-30 sobre los crudos por fuente). Se suma a eso que la descripción de
datos tiene que distinguir **dos cortes físicos con cadencias distintas**
(puntual Lun/Mie/Vie con intensidad propia vs compósito de turno sin ella), que
es lo que justifica los dos protocolos de evaluación. Parte de la reconciliación
pendiente.

## 13. Sesión 2026-10-01 (2) — a qué velocidad cambia el mineral

Pregunta del usuario: *el mineral cambia cada semana o dos semanas, y entonces
las intensidades también; compruébalo*. Se confirma en lo esencial y se corrige
un matiz que importa para el diseño del gate. Reproducible con
`scripts/verificar_escala_temporal_regimen.py` (solo lectura).

### 13.1 El régimen semanal: sí cambia cada 1-2 semanas

64 semanas con ≥ 50 lecturas (2025-07 → 2026-09), grupo dominante por semana:

| | |
|---|---|
| cambia de dominante de una semana a la siguiente | **54 %** |
| duración de una racha | mediana **1 sem**, media **1.8**, máx 6 |
| rachas de 1-2 semanas | **77 %** |
| pureza del dominante | mediana 0.59 |

### 13.2 Las intensidades se mueven con él

Varianza de cada feature que explica la semana (lecturas válidas, n = 24 461):

| feature | var. explicada | medianas semanales | sd intra-semana |
|---|---|---|---|
| `lr_mo_cu` | **0.52** | −3.30 … −2.11 | 0.23 |
| `lr_fe_cu` | 0.38 | −0.20 … +0.02 | 0.05 |
| `lr_zn_cu` | 0.37 | −5.52 … −4.90 | 0.15 |
| `logSumI` | 0.67 | +10.95 … +11.34 | 0.06 |

En `lr_mo_cu` el recorrido entre semanas es **~5×** la dispersión dentro de la
semana: la semana es un agrupamiento real de las intensidades, no ruido.

### 13.3 El matiz: la escala propia es 3-5 días, más rápida que semanal

Autocorrelación de la mediana diaria (415 días con ≥ 20 lecturas):

```
lag(d)  frac Cu  frac Mo  frac Zn  frac Fe  lr_fe_cu  lr_zn_cu  lr_mo_cu  logSumI
   1      0.65     0.69     0.66     0.57      0.70      0.65      0.76     0.87
   3      0.28     0.36     0.30     0.27      0.37      0.21      0.42     0.74
   5      0.06     0.16     0.13     0.18      0.23      0.05      0.24     0.71
   7      0.06     0.10     0.04     0.04      0.17     -0.03      0.15     0.68
  14      0.13    -0.05    -0.03    -0.09      0.06     -0.02     -0.11     0.67
  30     -0.01    -0.06    -0.14     0.01      0.03     -0.01     -0.10     0.59
```

Mineral y log-cocientes se descorrelacionan hacia el **día 5-7**. La racha
semanal de §13.1 es el mismo fenómeno visto con una lupa más gruesa: a
resolución semanal *parece* durar 1-2 semanas porque la agregación suaviza, pero
el recambio ocurre cada 3-5 días.

**`logSumI` es la excepción, y es informativa:** sigue correlacionado 0.59 a 30
días. Esa es la **deriva lenta de dilución/instrumento**, no el mineral.
Quedan separadas dos escalas de tiempo dentro de la misma señal — mineral rápido
(3-5 d) y dilución lenta (semanas-meses) —, lo que respalda que `dil` se
construya con una EWMA de 90 d y los log-cocientes no.

**Corrección a §2.6.** La frase *"no hay regímenes de semanas visibles en el
espectro"* estaba mal leída: sus números se reproducen (lag7d 0.04-0.17), pero
una autocorrelación nula a 7 días es **exactamente lo que produce un régimen de
3-5 días**, no su ausencia; y el espectro habla de *periodicidad*, no de
*persistencia*. Anotado en el propio §2.6.

### 13.4 Consecuencia para el gate de la arquitectura nueva

Fija la constante de tiempo que faltaba. El gate no puede operar:

- **por lectura** (~21 min): la etiqueta cruda cambia en el 11.7 % de las
  lecturas consecutivas y el **41 %** de las rachas dura una sola lectura
  (mediana 42 min) — chattering puro, y cada conmutación inyecta un escalón
  artificial en la estimación;
- **por semana:** llegaría tarde, el régimen ya cambió.

La escala del fenómeno es **3-5 días** → EWMA con *half-life* de ~1-2 días
(≈ 70-140 lecturas) más histéresis. Es un parámetro **derivado de los datos**.

### 13.5 La purga: una hipótesis que se planteó, se midió y resultó FALSA

**Hipótesis (descartada).** Si los features se descorrelacionan en 3-5 días, la
purga de 12 h de `split_purgado` no separaría nada relevante: las muestras de la
BD están a **2.9 d** de mediana (≤ 3 d en el 56 % de los pares) y sus leyes
consecutivas correlacionan `pFe +0.50 · pMo +0.46 · pCu +0.24 · pZn +0.01`, así
que una muestra de test podría estar a 2 días de su vecina de train compartiendo
régimen. Se predijo que con purga de 3-5 días se eliminarían filas y los R²
bajarían.

**Medido — no ocurre.** Barriendo la purga en la etapa 5 completa:

```
purga      pFe              pCu              pMo              pZn
 12h   -0.1703 (285)    0.5450 (285)    0.7758 (285)    0.8981 (215)
  3d   -0.1703 (285)    0.5450 (285)    0.7758 (285)    0.8981 (215)
  5d   -0.1696 (285)    0.5450 (285)    0.7701 (285)    0.8977 (215)
  7d   -0.1782 (285)    0.5447 (285)    0.7701 (285)    0.8977 (215)
```

Los R² no se mueven más allá de la cuarta cifra. Filas purgadas: 0 hasta 4 d,
**1** a 5 d, **4** a 7 d.

**Por qué — y esto es mejor noticia que la hipótesis.** Los cinco cortes del
walk-forward caen por sí solos en huecos de muestreo grandes:

```
fold 1  2023-06-14 -> 2023-06-19   hueco  4 d 18 h
fold 2  2024-02-09 -> 2024-02-19   hueco 10 d  5 h
fold 3  2024-08-07 -> 2024-08-12   hueco  5 d  2 h
fold 4  2025-04-11 -> 2025-04-16   hueco  5 d  0 h
fold 5  2025-10-15 -> 2025-10-22   hueco  7 d  6 h
```

Todos ≥ 4 d 18 h, o sea **iguales o mayores que la escala de descorrelación del
régimen** (3-5 d, §13.3). La fuga que se hipotetizó **no existe en estos folds**:
no hay ninguna muestra de train a menos de ~5 días del bloque de test, de modo
que la purga no tiene nada que quitar. El 23 % de los 341 intervalos supera los
5 días, y los 5 cortes cayeron en intervalos de ese tipo.

**Consecuencia para la sustentación — el argumento mejora.** Hasta ahora la purga
se defendía solo por corrección metodológica (tolerancia de emparejamiento 6 h,
ventana de evaluación 12 h). Ahora hay un argumento **positivo y medido**: los
huecos reales train→test (4.8 a 10.2 d) **exceden la escala de autocorrelación
del proceso** (3-5 d), así que los folds son efectivamente independientes. Eso es
mucho más fuerte que "la purga no elimina filas".

**El ADR del 2026-09-06 queda confirmado, no invertido:** la caída de R² de B2
viene del orden temporal y de los folds compartidos, no de la purga. La
corrección que este §13.5 intentó introducir **no procede**.

**Lo que sí queda en pie** de la autocorrelación entre muestras vecinas: el **N
efectivo es menor que 342**, porque muestras separadas 2-3 días comparten
información (Fe +0.50, Mo +0.46). Eso no sesga el R² al alza por fuga, pero sí
resta potencia a cualquier comparación entre particiones — razón de más para que
el control aleatorio siga siendo obligatorio, como ya lo es. Única salvedad
menor: el hueco del fold 1 (4 d 18 h) queda justo por debajo de los 5 d, donde
la autocorrelación aún es 0.06-0.24.

### 13.6 Siguiente

La purga queda cerrada sin cambios (`PURGA = "12h"` se mantiene, ahora con
justificación más fuerte). Despejado el camino para construir la arquitectura
nueva: 12 modelos (4 grupos × Cu/Fe/Mo) con gate de histéresis sobre EWMA de la
probabilidad del GMM, *half-life* 1-2 días (§13.4), entrenados sobre las 589
ventanas de compósito de train. Sigue siendo obligatorio que supere el control
aleatorio de particiones.

## 14. Sesión 2026-10-01 (3) — la arquitectura de 12 modelos por grupo, medida

Encargo del usuario: reemplazar la tabla de ruteo por una arquitectura donde
**cada grupo tiene sus propios modelos de Cu, Fe y Mo** y un **trigger** conmuta
de modelo ante cada lectura nueva. Se midió antes de construir el trigger.
Código: `src/pipeline/gate_grupos.py` (`python -m src.pipeline.gate_grupos`).

### 14.1 Por qué sobre ventanas de compósito y con purga de días

Con las muestras puntuales de la BD la arquitectura **no se puede ni entrenar ni
validar**: dos de los cuatro grupos no llegan a N≥60 (Mo 54, Zn 53) y el test
supervisado son 18 muestras con **Fe = 0**. Las ventanas de compósito dan **589
de train** (Cu 228 / Mo 176 / Zn 117 / Fe 68) y **153 de test**, y el compósito
mide exactamente las tres leyes del encargo. Pureza del grupo dominante: mediana
0.88, ≥ 0.7 en el 72 % de las ventanas.

**La purga aquí sí muerde**, al contrario que en §13.5: las ventanas son
contiguas cada 12 h, así que en cada corte hay una ventana de train pegada al
bloque de test, y el régimen se descorrelaciona en 3-5 d (§13.3). Se midió con
purga 12 h y 5 d; las conclusiones no cambian (las cifras de abajo son con 5 d).

### 14.2 Filtro 1 — control de particiones aleatorias (200, dentro de train)

Pasan 4 de 12 celdas: `pFe`/Zn (+0.678, pctl 100), `pCu`/Zn (+0.347, pctl 98),
`pMo`/Cu (+0.035, pctl 100), `pMo`/Fe (+0.119, pctl 100).

**Pero hay que leer el signo del R², no solo la mejora.** El global sobre
ventanas es **negativo** en Fe (−0.130) y Cu (−0.199): peor que predecir la
media. La "mejora" de `pFe`/Zn es pasar de −0.806 a −0.128 — de catastrófico a
malo, **sigue siendo negativo**. Presentar eso como un éxito habría sido
engañoso. Mo es el único con global sano: **0.847** fuera de fold.

### 14.3 Filtro 2 — supervivencia fuera de muestra (153 ventanas de test)

```
ley   grupo  N_tr  N_te   R2 glob   R2 loc   mejora
pFe   Zn      117    49    -1.171   -2.423   -1.252   <- COLAPSA
pCu   Zn      117    49    -0.021    0.033   +0.054   <- nada (ambos ~0)
pMo   Cu      228    37     0.597    0.700   +0.103   <- SOBREVIVE
pMo   Fe       68    12     0.481    0.476   -0.005   <- no se sostiene (N=12)
```

De las 4 que pasaron el control, **solo una sobrevive**: `pMo` en el grupo Cu, y
además crece fuera de muestra (+0.035 en train → +0.103 en test). `pFe`/Zn se
invierte por completo: era sobreajuste, no señal.

### 14.4 Veredicto: la arquitectura de 12 modelos no se sostiene

**1 de 12 celdas** sobrevive los dos filtros. Probando 12 celdas, que una pase es
lo que produce el azar; los dos filtros independientes la hacen más creíble que
un test suelto, pero **1 de 12 es una pista, no un resultado**. No se puede
replicar más: el período de producción tiene solo 8 ventanas del grupo Cu.

Razón de fondo, ya visible en los N: la arquitectura obliga a cada modelo de
grupo a aprender de **1/3 a 1/8** de los datos (228/176/117/68 contra 589). Con
estos tamaños el costo en varianza se come el beneficio de especializar, salvo en
la celda que sobrevive.

**Es la tercera vez que la idea de particionar por grupo cae con el mismo
control**: seis pruebas en §8-9, la etapa 5 sobre muestras de la BD (§11.1), y
ahora sobre ventanas de compósito con 8× más datos de test. Que falle con más
datos y mejor protocolo es, para la tesis, un resultado negativo **sólido y
defendible**, no una ausencia de resultado.

**El trigger no se implementó.** Construir un gate de histéresis para conmutar
entre modelos que no se distinguen del azar sería decoración; su diseño queda
especificado en §13.4 por si alguna vez hay evidencia que lo justifique.

### 14.5 Subproducto aprovechable

Sobre ventanas de 12 h el **modelo global de Mo** da **R² 0.847 fuera de fold** y
0.76-0.80 en test por grupo — mejor que la ruta por lectura de la etapa 5
(0.78). Refuerza el encuadre "Mo como resultado principal" y es una cifra usable.

Y queda confirmado por una vía más que **Fe y Cu no funcionan**: a nivel de
ventana su global es negativo, con o sin partición por grupo.

## 15. Sesión 2026-10-01 (4) — cabeza y relave: tipos de mineral donde corresponde

El usuario señaló que en la BD y en el PI hay datos de **cabeza (rebose de
hidrociclones)** y **relave final**, con su propio compósito cortado antes de los
rayos X. Resultó ser el hallazgo más importante de la sesión: los tipos de
mineral se venían definiendo sobre el **concentrado**, que es el producto de la
flotación, no el mineral que entra.

### 15.1 Lo que había sin extraer

Buscando en el PI aparecieron tres familias que nadie había mirado:

| Tags | Stream | Leyes | Historia |
|---|---|---|---|
| `7100AIP011-014MAN` | **cabeza** (rebose hidrociclones) | Fe, Cu, **Zn**, Mo | 2022-12-30 → hoy |
| `7100AIP031-034MAN` | **relave final** | Fe, Cu, Mo | 2022-12-30 → hoy |
| `7100AIP101-104MAN` | concentrado colectivo (ya en uso) | Fe, Cu, Ins, Mo | 2022-12-30 → hoy |
| 66 tags `_296290_*Canal*` | intensidades de **todas** las líneas de ambos couriers | 5 canales c/u | varía |

Los tres compósitos comparten grilla (07:30 / 19:30), historia y la propiedad de
§12.1: **se cortan antes de los rayos X**, así que ninguno tiene lectura de
intensidad propia. **La cabeza es la única que trae %Zn.**

Extraídos a `data/raw/composito_cabeza_pi.csv` y `composito_relave_pi.csv`:
**2603 filas cada uno, 0 nulos, 0 ceros centinela.** Intensidades de cabeza
(`_296290_RebHidroL1/L2_Canal*`) existen desde **2026-03-01** — 7 meses, frente a
las 58 épocas que dejaron el §8.3 inconcluso.

**Se explica el viejo misterio** de que ninguna de las otras 24 líneas calzara
con el PI (§10.3): no calzaban porque **cada stream tiene sus propios tags** y
solo se extraían los de `ConcFinal`.

### 15.2 Dos fuentes independientes de cabeza concuerdan

Compósito de cabeza del PI contra la ley de mina de `mineral.csv` (§14 previo),
calce 1:1 por turno, n=335: **Fe +0.677, Mo +0.676, Zn +0.581, Cu +0.575**, con
medias cercanas (Cu 0.486 mina vs 0.522 planta; Mo 0.029 vs 0.034). Dos
laboratorios, dos puntos de muestreo, dos sistemas: se validan mutuamente.

### 15.3 La cabeza explica el concentrado

Compósito a compósito, misma ventana de 12 h, **n=2601 sobre 4 años**:

```
desfase    cu_cab->cu   fe_cab->fe   mo_cab->mo
  +0h        +0.300       +0.342       +0.848
 +12h        +0.262       +0.300       +0.758
 +24h        +0.268       +0.323       +0.646
```

**`mo_cab → mo` = +0.848.** El decaimiento con el desfase confirma que el lag 0 es
el correcto: ambos son compósitos de 12 h del mismo período y la residencia de
planta es menor que la ventana. Para Cu y Fe hay 0.30-0.34: débil, pero **no cero
y de naturaleza distinta a las intensidades** — es lo primero con señal para esos
dos elementos en todo el proyecto.

### 15.4 Definición de los tipos de mineral (`src/pipeline/tipos_mineral.py`)

**Por qué log-cocientes y no leyes absolutas.** El usuario advierte que la muestra
de relave viene con **mucha dilución de agua**. Medido: la mayor parte de la
varianza de las leyes absolutas es un **factor de nivel común**, no composición —
el CV cae de **0.671 a 0.094** en el Zn de cabeza y de **0.256 a 0.105** en el Fe
de relave al pasar a log-cocientes. El alr de Aitchison contra Cu es invariante a
cualquier efecto multiplicativo de nivel, así que el tipo queda definido por la
composición y no por cuánta agua traía la muestra. Misma decisión que la etapa 2.

Protocolo idéntico al de la etapa 4, para que los resultados sean comparables:
ajuste **solo en train** (≤ 2026-05-31, n=2370), StandardScaler sin Yeo-Johnson,
GMM `full`, y **k por estabilidad ARI entre mitades disjuntas**, nunca por BIC.

**CABEZA — k=2** (ARI 0.769, silhouette 0.267; k=3 cae a ARI 0.506):

> **REVISADO el mismo día a k=3 — ver §17.** La regla automática (máxima
> estabilidad ARI) da k=2, pero k=2 **fusiona Cu y Mo**, que el ensayo
> independiente de la mina distingue con p=1.4e-15, y la operación necesita
> separarlos para conmutar estrategia de control. La tabla de abajo es la de
> k=2 y se conserva como registro; **el k vigente es 3** (`config.K_TIPO_CABEZA`).

| tipo | n | % | %Cu | %Fe | %Zn | %Mo |
|---|---|---|---|---|---|---|
| **Cu** | 1617 | 62.1 | **0.606** | 2.543 | 0.006 | **0.033** |
| **Fe** | 986 | 37.9 | 0.498 | **3.156** | 0.007 | 0.020 |

**RELAVE — k=3** (ARI 0.659; k=2 es inestable, ARI **0.069**):

| tipo | n | % | %Cu | %Fe | %Mo |
|---|---|---|---|---|---|
| **Mo** | 1045 | 40.1 | 0.049 | 2.108 | 0.007 |
| **Fe** | 796 | 30.6 | 0.046 | **2.609** | 0.003 |
| **Cu** | 762 | 29.3 | **0.061** | 1.962 | 0.005 |

> **Ojo con la lectura del relave:** son tipos de *lo que se pierde*, no de
> riqueza de mineral. El tipo "Cu" del relave es cola **relativamente rica en
> Cu** (0.061 contra 0.046), o sea **peor recuperación**, no mejor mineral.

### 15.5 Validación externa de los tipos de cabeza

**(a) Ancla geológica — la fase de la mina** (n=335):

| tipo cabeza | F05 mediana | F06 mediana | %Fe mina | %Mo mina |
|---|---|---|---|---|
| Cu | **67.1 %** | 32.7 % | 2.917 | 0.029 |
| Fe | 55.7 % | **44.2 %** | 3.381 | 0.019 |

Mann-Whitney sobre F05: **p = 7.4e-07**. El tipo de mineral en cabeza está
anclado a **dónde estaba cargando la pala**. F05 → tipo Cu, F06 → tipo Fe.

**(b) Los grupos del concentrado** (GMM sobre intensidades, n=803 ventanas):

| tipo cabeza | grupo Cu | grupo Mo | grupo Zn | grupo Fe |
|---|---|---|---|---|
| Cu | 37.4 % | **43.3 %** | 11.7 % | 7.6 % |
| Fe | 21.4 % | 8.9 % | **48.9 %** | 20.8 % |

Cabeza tipo Cu → concentrado Cu+Mo (**80.7 %**); cabeza tipo Fe → concentrado
Zn+Fe (**69.7 %**). Dos clusterings, sobre streams distintos y con instrumentos
distintos (compósito de laboratorio vs intensidades de rayos X), concuerdan: se
validan mutuamente. Hallazgo lateral: la cabeza Fe produce sobre todo el
concentrado que el GMM etiqueta "Zn".

**(c) Metalurgia — recuperación por fórmula de dos productos:**

| tipo cabeza | n | Rec. Cu | Rec. Mo | %Cu conc | %Mo conc | %Fe conc |
|---|---|---|---|---|---|---|
| Cu | 1617 | **91.32** | **83.06** | 26.17 | **1.39** | 28.18 |
| Fe | 984 | 90.47 | 80.35 | 25.17 | 1.01 | **29.20** |

El tipo de cabeza **se propaga al producto**: alimentación tipo Cu da concentrado
más rico en Cu y Mo y mejor recuperación. Cruzando con el tipo de relave, la
recuperación de Cu va de **88.54** (cabeza Fe + relave Cu, lo peor) a **92.73**
(cabeza Cu + relave Fe, lo mejor): **4 puntos** de recuperación, operativamente
relevante.

Y los dos extremos del circuito concuerdan: cabeza Fe → relave Fe en el 58.6 % de
los casos; cabeza Cu → relave Mo/Cu en el 86.5 %.

**(d) Persistencia:** rachas del tipo de cabeza de mediana **1 día**, media
**2.7 días**, máximo 26 días; 53 % duran ≤ 1 día, 25 % ≥ 3 días. Coincide con la
escala de descorrelación de 3-5 días del §13.3 — misma física, medida ahora
sobre la alimentación en vez del producto.

### 15.6 Qué desbloquea

- **§8.3 deja de estar inconcluso:** hay 7 meses de intensidades de cabeza
  solapadas con las de concentrado, no 58 épocas.
- **El Zn deja de ser inauditable** por el lado de la alimentación: la cabeza
  trae %Zn y el concentrado no.
- **Balance metalúrgico completo** a 12 h por 4 años: recuperación calculable.
- **Cu y Fe tienen por primera vez una variable con señal** (0.30/0.34 desde la
  cabeza), independiente de las intensidades.

### 15.7 Correcciones que esto obliga

- `CLAUDE.md` daba a entender que el courier solo expone `ConcFinal`: hay 66 tags
  de intensidad por stream.
- `works4cdp_assayspsi` existe en la BD con el esquema completo de sólidos y
  granulometría (`psolid`, `aSol`, `P80`, mallas) y está **vacía (0 filas)**. El
  hueco de `pSol` de alta frecuencia no es de diseño: es de poblado.
- La BD tiene cabeza en `sample_id` 17 (L1) y 18 (L2), con `pSol` (215 y 243
  muestras), pero solo 36 y 94 filas posteriores al inicio del PI de alta
  frecuencia: la vía BD sigue siendo insuficiente, la vía PI no.

## 16. Sesión 2026-10-01 (5) — ¿la cabeza mejora la estimación del concentrado? No

Medición directa de la pregunta que abrió el §15. Código:
`src/pipeline/aporte_cabeza.py` (`python -m src.pipeline.aporte_cabeza`).

**Dos escenarios, y la distinción importa más que los números.** El compósito de
cabeza de **la misma ventana** llega junto con el del concentrado: si lo tienes,
ya tienes el laboratorio, así que mide información pero **no es usable en línea**.
El de la **ventana anterior** (12 h) sí está disponible al momento de estimar y es
el único con valor operativo.

Base = las 5 features de intensidad del concentrado (`FEATS_REGRESION`);
tratamiento = base + las 4 leyes de cabeza. Mismos puntos fuera de fold,
`split_purgado` con purga de 5 d, y los dos filtros del §14.

### 16.1 Filtro 1 — walk-forward en train, control de 200 permutaciones

```
                         R2 base   R2 +cab   mejora  p95 ctrl  pctl  veredicto
lag 0   pFe                -0.130   -0.198   -0.069    -0.000     4  no aporta
        pCu                -0.199   -0.192   +0.008    +0.091    56  no aporta
        pMo                 0.847    0.849   +0.002    -0.002   100  no aporta
lag 1   pFe                -0.134   -0.170   -0.036    +0.003    24  no aporta
        pCu                -0.231   -0.239   -0.008    +0.118    26  no aporta
        pMo                 0.847    0.841   -0.006    -0.001    69  no aporta
```

### 16.2 Filtro 2 — fuera de muestra (test)

```
        pFe    -0.198 -> -0.295  (-0.097)   |  lag 1:  -0.201 -> -0.217  (-0.016)
        pCu    -0.115 -> -0.046  (+0.069)   |  lag 1:  -0.115 -> +0.037  (+0.152)
        pMo     0.827 ->  0.867  (+0.040)   |  lag 1:   0.831 ->  0.837  (+0.006)
```

### 16.3 Veredicto: la cabeza no aporta, y se entiende por qué

**Ninguna celda pasa el filtro 1 en ninguno de los dos escenarios.**

La explicación es la parte interesante: §15.3 midió `mo_cab → mo` = **0.848**, y el
modelo que usa **solo** las intensidades del concentrado ya llega a R² **0.847**.
Son el mismo número. **La información que la cabeza tiene sobre el Mo del
concentrado es redundante con lo que los rayos X ya ven** — la cabeza está aguas
arriba de la misma señal, no es información independiente sobre el producto. Para
Cu y Fe no hay nada que mejorar: el modelo base es negativo, peor que la media.

**El único aparente éxito es falso por el mismo patrón del §14.** `pCu` a lag 1
mejora +0.152 en test (−0.115 → +0.037) pero **falla el filtro 1** (percentil 26).
Es la figura espejo del `pFe`/grupo Zn del §14.3, que pasaba en train y colapsaba
fuera de muestra. Un resultado que solo aparece en uno de los dos filtros no se
reporta como resultado.

### 16.4 Lección metodológica — el control necesita un piso de magnitud

En la primera corrida, `pMo` a lag 0 salió **"APORTA"** con una mejora de
**+0.002**. El control por permutación añade 4 columnas de ruido, así que su p95
queda en ~0 o negativo y **cualquier mejora no negativa lo supera**. La vara del
control es necesaria pero no suficiente: hace falta además un **piso de
magnitud**. Se añadió `MEJORA_MIN = 0.01` y el veredicto pasó a "no aporta", que
es lo honesto. Aplicable a cualquier prueba de feature nueva en este proyecto,
no solo a esta.

### 16.5 Lo que queda sin probar, y es la mejor apuesta

Esto cierra la cabeza **como feature de regresión del compósito**, no la cabeza
en general. Siguen abiertas dos vías distintas:

1. **Las intensidades de cabeza** (`_296290_RebHidroL1/L2_Canal*`, desde
   2026-03-01, cadencia de courier): son una señal **aguas arriba y en tiempo
   real**, no un ensayo de laboratorio que llega tarde. Es otra proposición y es
   la que desbloquea el §8.3.
2. **La cabeza como variable de régimen / gating**, no como feature de
   regresión — que es para lo que el §15.5 mostró que sirve (ancla geológica,
   p=7.4e-07, y correspondencia con los grupos del concentrado), y que es
   distinto de predecir la ley punto a punto.

## 17. Sesión 2026-10-01 (6) — k=3 en cabeza, y el reporte de producción ago-sep

Encargo del usuario: validar con la cabeza los **4 grupos** que dio el
concentrado, comparar con `mineral.csv`, y con agosto-septiembre como producción
hacer un **reporte gráfico** de cómo afectó el cambio de mineral a las variables
operativas de los espesadores, con énfasis en el nivel de interface.

### 17.1 Los 4 grupos del concentrado NO se reproducen en la cabeza

Métricas del GMM sobre log-cocientes de cabeza (train n=2370):

```
 k    bic   silhouette   ARI estab.   grupo_min_%
 2  18990     0.267        0.769         39.1
 3  18947     0.209        0.506         28.6
 4  18960     0.240        0.561          8.9
 5  18990     0.202        0.365          8.7
 6  19002     0.223        0.446          5.6
```

Dentro del rango que la operación espera (3 o 4), **k=4 gana en las métricas**
(silhouette 0.240 vs 0.209, ARI 0.561 vs 0.506) **pero no sobrevive el contraste
con la mina**: sus tipos Cu-2 y Zn+Fe son **indistinguibles** para el ensayo de
mina (%Fe 3.360 vs 3.369) y el cuarto grupo cae a **8.9 %** (n=24 en el traslape
con `mineral.csv`); además el "cuarto grupo" es un **desdoble de Cu**, no un
mineral nuevo. Con **k=3** cada tipo tiene firma propia:

| tipo cabeza | n | % | %Cu | %Fe | %Zn | %Mo | F05 mina | %Mo mina |
|---|---|---|---|---|---|---|---|---|
| **Cu** | 976 | 37.5 | **0.605** | 2.679 | 0.005 | 0.024 | 59.1 | 0.021 |
| **Mo** | 882 | 33.9 | 0.578 | 2.546 | 0.007 | **0.041** | **67.8** | **0.032** |
| **Fe+Zn** | 745 | 28.6 | 0.488 | **3.187** | **0.008** | 0.020 | 55.8 | 0.020 |

Kruskal contra `mineral.csv` (n=335): **%Mo p=1.4e-15**, %Fe p=1.1e-07,
%Cu p=9.3e-07, F05 p=5.9e-05. Los tres tipos se validan contra un ensayo
independiente y contra la fase de la pala.

**Hallazgo de fondo:** en la cabeza **Zn y Fe no se separan** — se separan recién
en el concentrado. Los 4 grupos del concentrado son, entonces, **3 tipos de
mineral de alimentación + un efecto de la flotación**, no 4 minerales distintos.
Se fija `K_TIPO_CABEZA = 3` en config, con la justificación escrita ahí; la regla
automática de máxima estabilidad (que daría k=2) se imprime igual para que la
diferencia quede visible y no escondida.

### 17.2 Reporte de producción agosto-septiembre 2026

Extraídos del PI los 18 tags de `docs/espesador_desc.txt` a 5 min (llegaron 16;
faltan `x9`/`x10`, niveles de piscina) → `data/raw/esp_agosep_pi.csv`, 17 569
registros, 2026-08-01 → 2026-10-01. Agregado a las 119 ventanas de 12 h con tipo
de mineral. **El GMM se ajustó solo hasta 2026-05-31, así que este período es
fuera de muestra para el clustering.**
Código y figuras: `data4Espesadores/src/reporte_mineral_espesadores.py`,
`reports/fig_esp_01..03` + `mineral_vs_espesadores_agosep.csv`.

**Dos trampas encontradas y tratadas:**

1. **La línea L2 estuvo fuera del 19 al 31 de agosto** (`y5` ≈ 1.00 = agua).
   Agosto concentra 34 de las 41 ventanas Fe+Zn, así que sin enmascarar salía
   "Fe+Zn baja la densidad a 1.06" como hallazgo — y era el equipo parado.
2. **Confusión temporal:** agosto es Fe+Zn (34/41) y septiembre es Mo (45/68).
   Todo se reporta **centrado por mes**, que es lo que vale.

| variable | Cu (n=10) | Mo (n=68) | Fe+Zn (n=41) | p crudo | **p intra-mes** |
|---|---|---|---|---|---|
| **Nivel de interface (m)** | **1.872** | **2.167** | 2.129 | 0.031 | **0.0016 ✔** |
| Nivel cajón distribuidor (%) | 63.5 | 57.9 | 58.0 | 0.154 | 0.0065 |
| Flujo de ingreso (m³/h) | 1695 | 1449 | 1459 | 0.170 | 0.0086 |
| Nivel tanque TK001 (%) | 34.4 | 44.6 | 38.5 | 0.009 | 0.0135 |
| Nivel tanque TK002 (%) | 34.0 | 44.9 | 38.3 | 0.012 | 0.0148 |
| BedMass (t) | 60.8 | 60.4 | 60.5 | 0.018 | 0.041 |
| Densidad descarga L1 | 1.552 | 1.552 | 1.618 | 0.019 | 0.087 |
| Torque de rastra (%) | 30.9 | 32.5 | 32.2 | 0.116 | 0.088 |
| Flujo a piscinas (m³/h) | 1677 | 1761 | 1905 | 0.052 | 0.761 |
| Flujo de floculante (m³/h) | 0.329 | 0.345 | 0.336 | 0.473 | 0.837 |

**✔ = único que sobrevive Bonferroni** por las 10 variables probadas
(umbral 0.005). El **nivel de interface** es el efecto robusto; el resto es
sugerente. Y **el floculante no cambia** (p=0.84): la diferencia de interface no
se explica por dosificación distinta.

### 17.3 La interface como indicativo de recuperación de agua: depende de la escala

El usuario plantea que el nivel de interface es directamente proporcional a la
subida de los tanques temporales de agua, y por tanto indicativo de recuperación.
**Medido, y la respuesta es que depende de la escala de tiempo:**

```
corr(interface, nivel TK001)      por HORA      +0.377
corr(interface, SUBIDA de TK001)  por HORA      +0.070   <- la tasa no correlaciona
corr(interface, flujo a piscinas) por HORA      -0.206   <- signo contrario
corr(interface, nivel TK001)      por VENTANA DE 12 h   +0.525   <- se sostiene
```

**A escala de turno la relación se sostiene (r=+0.525)**, y el ordenamiento por
tipo de mineral la acompaña: Mo tiene la interface más alta (2.17 m) **y** los
tanques más altos (45 %); Cu la más baja (1.87 m) y los tanques más bajos (34 %).
A escala horaria no se sostiene, y el flujo a piscinas incluso va al revés —
esperable, porque ese flujo lo fija el bombeo, no la decantación.

**RESUELTO por el usuario (misma fecha):** el nivel de interface es la **altura
de agua clara por encima del material sedimentado**. No es la cama de sólidos,
que es lo que se había supuesto al escribir el párrafo anterior. Con eso:

- **interface alta = más agua clara = más recuperación de agua**, tal como
  planteaba el usuario, y todos los signos medidos van en la dirección correcta;
- la objeción física que se había levantado (*«interface alta implicaría menos
  rebose»*) **era incorrecta** y queda retirada;
- el flujo a piscinas (−0.206) **no contradice nada**: es una variable
  **manipulada** por el bombeo, no una medida de recuperación. Para recuperación
  hay que mirar el nivel de los tanques, que es lo que se acumula.

**Consecuencia sobre el §11 histórico:** allí se concluyó que el relave tipo Fe
*«decanta peor»* por tener +0.215 m de interface a igual floculante. **Es al
revés: decanta mejor.** Corregido en el ADR correspondiente.

### 17.4 Para el sistema experto

Con estos datos, el único efecto defendible del tipo de mineral sobre la
operación es el **nivel de interface** (p=0.0016 centrado por mes, fuera de
muestra), con los tanques de agua acompañando.

**Lectura con el signo ya confirmado** (interface = altura de agua clara):

| régimen | agua clara | nivel TK001 | lectura |
|---|---|---|---|
| **Mo** | **2.17 m** | **45 %** | decanta mejor, **más agua recuperada** |
| Fe+Zn | 2.13 m | 38 % | intermedio |
| **Cu** | **1.87 m** | **34 %** | decanta peor, **menos agua recuperada** |

Y **el floculante no cambia entre regímenes** (p=0.84): la planta **no está
compensando** el cambio de mineral, lo está absorbiendo como menos recuperación
de agua cuando entra mineral tipo Cu. Ahí está el margen del sistema experto:
anticipar el régimen Cu y subir floculante antes de perder el agua, en vez de
reaccionar. Es **una base para conmutar**, no
una ley: Cu tiene n=10 ventanas y además corre con más flujo de ingreso
(1695 vs 1449, p=0.0086), que es confusión por carga que estos dos meses no
permiten separar.

---

## 18. Sesión 2026-10-21 — los 5 canales son simultáneos: rediseño de la etapa 1a

Partió de una observación del operador sobre `courier_pi.csv` —"las
intensidades de los metales y la del `n6sc` tienen el mismo timestamp"— y
terminó rehaciendo la reconstrucción de lecturas del PI. Resultado neto: el
pipeline queda correcto y **ninguna conclusión de la tesis se mueve**.

### 18.1 El hecho del equipo, medido

El courier mide los 5 canales en **una sola lectura**, no canal por canal.
En modo automático el mínimo entre muestras es 2 min y rotan 10 líneas, así
que el concentrado final toca cada ~20 min; con menos líneas en servicio el
ciclo se acorta proporcionalmente. Verificado sobre los 413,834 eventos
crudos hasta 2026-09-30:

| medición | valor |
|---|---|
| instantes de cambio, por canal | `n1fe` 25,439 · `n2cu` 25,439 · `n3zn` 25,439 · `n4mo` 25,439 · `n6sc` 25,439 |
| eventos agrupados | 25,179, **los 25,179 con los 5 canales** |
| cambios de `n6sc` sin cambio de metales a < 10 min | **0** |
| spread del evento (primer vs último canal) | p50 = 0 s · p99 = 3 s · max = 145 s |
| intervalo entre lecturas | p50 = 20.9 min (p25 18.1, p75 25.5) |

Los cinco canales tienen **exactamente** el mismo número de cambios, y en
94.8 % de los eventos los 5 se archivan en el mismo segundo. No existe un
solo caso en 15 meses donde el Sc se mueva y los metales no.

**Consecuencia:** una fila del PI a la que le falta un canal **no es una
lectura**. Es media lectura archivada aparte. El desfase está cuantizado
(3, 4, 6-7, 14, 37-39, 63, 143-145 s) y cambia de signo: firma de un
mecanismo de escritura del PI, no de medición. En 24 de los 25 eventos con
spread > 3 s los 4 metales comparten el segundo exacto y el único corrido es
`n6sc`; el caso restante (2026-08-16) es `n2cu` corrido 63 s.

### 18.2 Los tres defectos de `limpiar_pi()`

La función definía la lectura por los 4 metales y después pegaba `n6sc` con
una ventana. Medido:

1. **`TOL_N6SC = 60s` no alcanza.** En los 5 eventos con el Sc a 143-145 s,
   el PI republica el Sc **viejo** en el hueco, así que el respaldo lo
   agarra: 4 lecturas se quedaron con el Sc de la muestra anterior y 1 en
   NaN.

   | lectura | Sc asignado | Sc correcto |
   |---|---|---|
   | 2026-03-13 09:28:55 | 1455.16125 | 1444.49829 |
   | 2026-03-16 02:25:51 | 1389.99792 | 1422.89038 |
   | 2026-03-29 10:42:15 | NaN | 1458.50300 |
   | 2026-04-01 18:29:23 | 1434.72253 | 1434.98645 |
   | 2026-07-09 13:52:28 | 1346.33911 | 1321.69482 |

2. **`TOL_EVENTO = 5s` agrupaba FILAS, no cambios de valor.** El PI archiva
   seguido una lectura nueva a menos de 5 s de la última republicación de la
   anterior; `agg(first)` se quedaba con la vieja y la nueva reaparecía en su
   republicación siguiente, **fechada ~100 s tarde**. Son **1,201 lecturas
   (4.7 %)**, no un caso aislado. Era invisible porque `sostenida_s` solo
   llega a 0 cuando la lectura tenía una sola republicación (14 casos).

3. **Una fila parcial podía fechar y valorizar una lectura.** Con `n2cu`
   archivado 63 s aparte (2026-08-16 17:37:28) salía una fila mezclando el
   `n2cu` nuevo con los otros tres arrastrados por `ffill` de la lectura
   anterior: una lectura que nunca existió. 2 casos (el otro, 2026-07-31
   11:48:21).

### 18.3 `lecturas_por_evento()` — el rediseño

Agrupa **instantes de cambio**, no filas, y trata los 5 canales por igual.
El corte no es temporal sino **estructural**: como cada canal cambia una vez
por lectura, ver dos veces el mismo canal significa que empezó otra. Eso
separa los ciclos cortos reales (hay lecturas legítimas a 35 s) sin partir
los eventos con 145 s de spread — algo que **ninguna tolerancia temporal
única puede hacer**, porque los dos rangos se solapan.

Una fila incompleta no se descarta: aporta sus canales al evento pero no
fecha ni valoriza la lectura por sí sola. Aplicar la regla como *filtro de
filas* en cambio cuesta 74,561 filas (18 %) y **1,290 lecturas**, justo las
del período de desfase.

Dos condiciones extra que el código viejo cumplía por accidente vía
`TOL_EVENTO`: un "cambio" cuyo valor es el que el canal ya tiene en la
lectura en curso, o el que tenía en la anterior, es **re-archivado** y no
abre lectura. El PI a veces revierte al valor viejo 1 s después de publicar
el nuevo y vuelve ~100 s más tarde (12 casos, p.ej. 2025-08-15 03:31:08 `Y` /
03:31:09 `X` / 03:32:48 `Y`). Son 60 instantes descartados.

### 18.4 El diff

| | `limpiar_pi` | `lecturas_por_evento` |
|---|---|---|
| lecturas | 25,429 | 25,427 |
| lecturas incompletas | 0 | 0 |
| mismas 5 intensidades en ambos | — | 25,422 |
| solo en el viejo | 7 (5 con Sc rancio + 2 quimeras) | — |
| solo en el nuevo | — | 5 (las mismas con Sc correcto) |
| hora corregida | — | 1,515 (de ellas 1,201 por −100 s) |
| calce BD ↔ PI | 125/125 | 125/125 |

Comparador reproducible: `notebooks/27_diff_reconstruccion_pi.py`.

### 18.5 Impacto aguas abajo: ninguno

Pipeline corrido completo (1a → 6) y `data/processed/` regenerado.

- **Asignación de grupo: 25,846/25,846 idénticas (100.00 %).** Cero
  reasignaciones.
- Métricas de la etapa 5 contra las 18 muestras de test: pFe 0.253,
  pCu 0.518, pMo 0.628, pZn 0.832 — iguales a §11.2.
- `models/anomaly_detector.joblib` salió **byte-idéntico**.
- 1c: 25,848 → 25,846 filas; anomalías 950 → 949; `validas_ley` 361 → 361.

La razón: **`n6sc` ya no entra en ninguna feature.** Desde que se eliminó la
ortogonalización (§10.6), el clustering usa log-cocientes entre metales y la
regresión usa fracciones de cierre más `logSumI`, que suma `METALES` y no
`CANALES` (`features.py:105`). El Sc no toca nada, y los metales cambiaron en
una sola lectura de 25,422.

Dicho de frente: esto fue **saneamiento de calidad de dato, no un hallazgo
que cambie conclusiones**. Lo que gana es cualquier análisis futuro que **sí**
use el tiempo —ventanas contra el compósito, lags, el corte train/test al
segundo—: las 1,201 lecturas mal fechadas por 100 s sí habrían afectado un
análisis de desfase temporal.

### 18.6 Dos cifras que estaban mal en esta bitácora

- **Las "18 muestras de la BD que no aparecen en el PI" (§10.3) no existen.**
  Intenté reproducirlas con tres criterios de calce distintos: dos dan
  125/125 y solo el más estricto (`dropna` sobre los 5 canales) falla, en
  **5** muestras — 2026-04-15, 04-29, 05-15, 05-18 y 07-01, todas con el Sc
  corrido 1-3 s. El commit que introdujo la frase (`b99d9f0`) es el mismo que
  implementó la 1a, y el script de merge anterior solo concatenaba, así que
  el diagnóstico pre-fix nunca se comiteó. **Además el mecanismo no cierra:**
  en una fila partida los 4 metales están completos, así que el calce por
  valor no se rompe. No usar ese número.
- **`unificacion.py` decía 122/122** donde lo medido es 125/125. Corregido.
- El "pico de 6,399 filas en mayo" (§10.2) son **filas crudas**, no lecturas:
  a nivel de lectura son 442 en mayo y 1,310 en total.

### 18.7 Lo que queda abierto, con número

- **533 lecturas a < 10 min de la anterior (2.1 %).** No son artefacto: los
  5 canales cambian en 531/533, y la magnitud del cambio es normal (mediana
  1.3-2.2 % vs 1.54 % de las normales). Vienen en rachas —la más larga 75
  lecturas en 5.7 h el 2026-06-12, con mediana 4.5 min— y el intervalo dentro
  de las rachas está cuantizado en los 2 min por muestra (modo en 4-5 min ≈
  2 líneas en el ciclo). **Son ciclos cortos reales**, no error.
- **7 lecturas con una predecesora a 35-67 s** (antes 12; el rediseño
  explicó 5). Con los 5 canales cambiando y `sostenida_s` normal. Por debajo
  de los 2 min no hay configuración de líneas que lo produzca. Hipótesis a
  zanjar con acceso al PI: el analizador emite un preliminar y un definitivo
  del mismo ciclo, o el PI escribe y corrige. Son 2026-01-23 23:42:49,
  2026-02-13 17:55:32, 2026-03-13 15:00:40, 2026-04-02 22:46:35,
  2026-07-27 14:45:50, 2026-07-30 06:58:08, 2026-08-05 09:34:30.
- **Una medición de señal baja**, no de archivado: 2026-04-02 22:45:34 con
  `n1fe` 3,935 contra ~30,000 normal y el `n6sc` **subiendo** a 2,058 — celda
  vacía o purga. La captura el IsolationForest de la 1c.

---

## 19. Sesión 2026-10-21 (2) — noviembre 2025 es límite de instrumento, y el Mo es un gradiente

Encargo: agrupar con intensidades crudas, sin correcciones, probando varios
algoritmos y métricas de validación. Terminó separando la deriva de la
magnitud en tres causas y cerrando la pregunta de los grupos de mineral en el
concentrado. **Dos resultados nuevos y un negativo replicado.**

Reproducible con:
`notebooks/28_experimento_clustering_intensidades_crudas.py`,
`notebooks/29_clustering_post_mantenimiento.py --desde X --hasta Y`,
`scripts/diagnostico_deriva_intensidades.py`, `scripts/comparar_epocas_mo.py`.

### 19.1 Lo que confirmó planta

Tres hechos, aportados por el usuario, que convierten hallazgos estadísticos
en hechos documentados:

- **El mantenimiento de noviembre 2025 existió**, y después se hizo una
  **campaña de muestreo nueva para rehacer las ecuaciones del Outocal**. Esa
  secuencia —mantenimiento, luego recalibración con muestras nuevas— es lo que
  se hace cuando la respuesta física cambió lo bastante para invalidar las
  ecuaciones viejas.
- **El insoluble no tiene canal físico de medición.** No hay hardware: lo que
  hay es estimación, y lo único real es el ensayo del laboratorio químico.
- **No existe tag de densidad en la línea de concentrado final.**

### 19.2 La deriva de la magnitud son TRES causas, no una

§2.3 tenía registrada una "deriva relativa entre trimestres de 22-25 %"
atribuida a decaimiento de fuente. Separadas (solo período pre-recalibración,
n=253, `logSumI ~ química + pSol + tiempo`):

| causa | tamaño | evidencia |
|---|---|---|
| **dilución de la muestra** | dominante: R² 0.221 → **0.696** al entrar `pSol`; +0.0104 de `logSumI` por punto de %sólido (~1 % de intensidad) | `pSol` t=+19.62 |
| **recalibración por canal** | escalón de −22 % a +4 % según canal, nov-2025 a dic-2025 | los cocientes entre canales se movieron −15.1 % a +14.1 %; una ganancia común no puede hacerlo |
| **decaimiento de fuente** | **−2.7 %/año**, constante | sobrevive controlando química *y* `pSol`: t=−6.01, p=6.7e-09 |

El escalón del `n6sc` es de un día (mediana diaria 1781 → 1503 → 1438 entre el
2025-12-09 y el 2025-12-11) y −18.9 %. Las fechas por canal están
**escalonadas** (n1fe ~2025-11-30, n6sc 2025-12-10), consistente con calibrar
canal por canal; `n3zn` y `n4mo` son demasiado ruidosos para fechar.

**La hipótesis de planta —líneas de agua agregadas contra espuma y atoro—
sobrevive el control de química**: `pSol ~ química + tiempo` deja el tiempo en
−2.00 puntos/año (p=1.6e-07) y manda todos los coeficientes de química a no
significativos (`pIns` de −0.33 a +0.02). O sea: el mineral cambió **y además**
la muestra se diluyó, son dos tendencias independientes. Sin tag de densidad no
hay forma de comprobarlo de modo independiente, así que queda **sostenida, no
demostrada**.

**Corrección a un dato de esta misma sesión:** la caída de `pSol` **no es
monótona**. Por trimestre: 2022Q4 25.3 · 2023Q2 24.7 · **2023Q3 31.0** ·
2023Q4 30.0 · 2024Q2 27.9 · 2025Q2 23.9 · 2026Q1 20.5 · 2026Q3 16.4. Hay un
máximo en 2023H2 y la caída arranca en 2024.

### 19.3 La invarianza a la dilución es física del canal, no una corrección

El agua absorbe los rayos X según la energía (~E⁻³), así que un cociente entre
canales de energía **parecida** es invariante a la dilución y uno entre
energías distintas no puede serlo. Los signos lo confirman:

```
feature     |dE| vs Cu   corr(.,pSol)   centrado por mes
logSumI              -         +0.721            +0.738
lr_zn_cu          0.59         -0.230            -0.216
lr_fe_cu          1.65         +0.519            +0.438
lr_mo_cu          9.43         -0.278            -0.215
```

Fe (6.40 keV) se absorbe **más** que Cu (8.05) → signo positivo. Mo (17.48) se
absorbe **menos** → negativo. Zn (8.64) ≈ Cu → §10.7 ya lo midió en t=−1.3
controlando química, o sea **cero**. `lr_zn_cu` es el invariante por
construcción; `lr_fe_cu` y `lr_mo_cu` no pueden serlo, y el techo de la
corrección actual (0.91 → 0.73 sd) es ese.

Y **dentro de época el proxy es mejor que el +0.659 global de §2.4**:
`corr(logSumI, pSol)` centrado por mes da **+0.840** (pre) y **+0.684**
(post). Lo degradaba mezclar épocas de calibración.

### 19.4 El k=2 de los 15 meses es la recalibración; dentro de época, la dilución

Siete algoritmos × k=2..8 más HDBSCAN sobre las 5 intensidades crudas
(18,596 lecturas de train, escalado ajustado solo en train):

- **Unanimidad en k=2**: silhouette 0.32-0.33 y estabilidad ARI 0.96-1.00 en
  KMeans, GMM, Ward, Spectral y SOM; HDBSCAN da 2 grupos con
  `min_cluster_size` 100, 300 y 1000. Hopkins 0.940.
- **Y ese k=2 es un calendario**: grupo 1 al 92-99 % hasta 2025-11, 21 % en
  2025-12, 0-10 % desde 2026-01. Las leyes del laboratorio son iguales en los
  dos grupos (pFe 28.28/28.25, pZn 0.17/0.17), `eta²` ≈ 0.
- **Dentro de cada época**, en cambio, el k=2 separa **sólido medido**:
  centrado por mes, +2.17 vs −2.58 (p=0.0009) en pre y +0.79 vs −1.92
  (p=0.0220) en post. Esos sí son los dos niveles de dilución.

### 19.5 El autoencoder no es reproducible, y el patrón de su fallo es el hallazgo

Un MLP 5-3-2-3-5 entrenado X→X (sin `torch` en el entorno) separa por su
cuenta magnitud de composición: una unidad del cuello queda con |corr| 0.73-0.97
contra `logSumI` y la otra con los log-cocientes. Redescubre lo que el alr
impone por construcción. Pero **la partición no se sostiene entre semillas**:

| ventana | k=2 | k=3 | k=4 |
|---|---|---|---|
| 15 meses (notebook 28), ARI mediano | — | +0.176 | +0.230 |
| pre (notebook 29), ARI mediano | +0.120 | +0.340 | +0.389 |
| post (notebook 29), ARI mediano | −0.002 | +0.284 | +0.274 |

Y **el MSE no indica qué semilla salió bien**. En los 15 meses (notebook 28,
k=4) la semilla 33 tiene el **mejor** MSE de las ocho (0.236) y el peor
comportamiento (`eta²` 0.046, `AMI_mes` 0.377), mientras la 13 con MSE 0.239 da
`eta²` 0.205 y `AMI_mes` 0.077. En la ventana post pasa lo mismo: las semillas
42 y 7 dan `c_mo` 0.532 y 0.526 y la 1 da 0.013. Elegir semilla por `eta²`
sería usar las leyes del laboratorio para escoger un modelo no supervisado.

**Pero 5 de 8 semillas dan `c_mo` entre 0.48 y 0.64 con ARI de 0.25 entre
ellas**: encuentran el mismo eje y lo cortan en lugares distintos. Eso solo
pasa con un gradiente.

### 19.6 El Mo es un GRADIENTE, no grupos — replicado en dos épocas

Prueba decisiva: la variable continua contra la etiqueta discreta, sobre el
compósito y en las dos ventanas.

```
                              pre (jul-nov 2025)    post (dic 2025-sep 2026)
compositos usables                      243                     524
corr(lr_mo_cu, mo)                   +0.816                  +0.877
R2 CONTINUA                           0.667                   0.769
Spearman                             +0.917                  +0.903

R2 de la etiqueta discreta (mejor semilla del AE), mismo Mo:
  k=2                                 0.486                   0.543
  k=3                                 0.547                   0.552
  k=4                                 0.529                   0.599
  k=6                                 0.582                   0.610
  k=8                                 0.620 (post) / 0.587 (pre)
```

**Ninguna partición, con ningún k, en ninguna de las dos ventanas, le gana a la
variable continua.** Discretizar cuesta 0.15-0.23 de R². Y `lr_mo_cu` no ajusta
ningún parámetro: ese 0.77 no tiene nada que sobreajustar.

Cinco líneas de evidencia convergentes, cada una replicada:

| observación | implica |
|---|---|
| la continua gana a toda etiqueta | discretizar destruye información |
| ARI entre semillas 0.25 con `c_mo` similar | el corte es arbitrario |
| silhouette ≤ 0.31 en todo k y algoritmo | no hay grupos separados |
| HDBSCAN: 100 % ruido en pre; 38.8 % y nada con `mcs`≥300 en post | no hay densidades separadas |
| el histograma de `lr_mo_cu` tiene **una** joroba, sin valles | es un continuo |

**Fe, Cu e insoluble: nada por esta vía, en las dos ventanas.** `lr_fe_cu` vs
`fe` R²=0.069; vs `cu` R²=0.092 (post) y 0.206 (pre); vs `ins` R²=0.000.
`c_cu` y `c_ins` ≤ 0.07 en todas las combinaciones. Coherente con 19.1: el
insoluble no tiene canal.

### 19.7 Las dos épocas se pueden juntar para Mo, con offset

```
regresión  mo ~ lr_mo_cu  por época
  pre : pendiente +1.9945  intercepto +7.0442  R2 0.669  RMSE 0.4809
  post: pendiente +1.9520  intercepto +6.7346  R2 0.769  RMSE 0.3449

modelo conjunto con dummy e interacción
  post (salto)                  -0.3097  p=0.203     <- no significativo
  lr_mo_cu x post (pendiente)   -0.0425  p=0.641     <- no significativo

validación cruzada de época
  ajuste en pre  -> post: R2 +0.692  sesgo +0.199
  ajuste en post -> pre : R2 +0.613  sesgo -0.196
```

El salto estimado **coincide con lo que predice la recalibración medida
aparte**: el cociente `n4mo/n2cu` cambió +14.1 %, o sea `lr_mo_cu` se corrió
log(1.141)=+0.132, que por la pendiente 1.95 da −0.257 de intercepto; medido,
−0.31. Dos mediciones independientes que se encuentran.

**Recomendación: juntar las épocas con offset de época.** Un parámetro, y
elimina el sesgo de ±0.20 que se ve en la predicción cruzada.

### 19.8 Lo que esto explica hacia atrás

Da el mecanismo de cuatro resultados que el proyecto tenía medidos sin
explicación:

- **Silhouette bajo 0.3 en todo k** (§10.8): un continuo no tiene grupos.
- **Ningún modelo local pasó nunca el control aleatorio** (ADR 2026-09-30): los
  grupos son rebanadas arbitrarias de un gradiente, así que un modelo por grupo
  ajusta ruido. No era falta de muestras.
- **Solo el Mo es defendible**: es el único con gradiente fuerte.
- **Las métricas internas engañan en este problema**: premian la partición por
  dilución (silhouette 0.25-0.31) y castigan la de composición (0.07-0.16),
  porque miden distancias en el espacio crudo donde la magnitud domina la
  varianza. Las únicas que apuntaron bien fueron la validación externa contra
  el compósito y el control de época.

### 19.9 Consecuencias y abiertos

- **Defecto a corregir en la etapa 3:** `dil = logSumI − EWMA(90 d)` atraviesa
  la ventana de recalibración de nov-2025 a ene-2026, así que durante ~3 meses
  absorbe un escalón de instrumento como si fuera dilución. El decaimiento de
  −2.7 %/año sí lo maneja una EWMA; el escalón no. Esto **sí mueve resultados**:
  `dil` escala Fe/Cu.
- **Para el sistema experto (objetivo 2):** la variable no es una etiqueta de
  grupo con histéresis, es `lr_mo_cu` suavizado con EWMA (half-life 1-2 días,
  §13) con umbrales sobre el valor continuo. El umbral lo fija metalurgia según
  a partir de qué %Mo conviene conmutar: no es algo que el clustering descubra.
- **Lo de mayor valor sin usar: la campaña del Outocal.** No está en
  `works4cdp_assay` — el muestreo sigue plano en 5-13 por mes sin ningún pico
  tras noviembre. Son pares intensidad↔ensayo densos y controlados en el estado
  de instrumento vigente, contra las 64 muestras de laboratorio que tiene esa
  ventana hoy.
- **Limitación para la tesis:** estos escalones solo son detectables desde el
  2025-07-13, cuando arranca la historia del PI. Antes hay 348 muestras de la
  BD a 5-13 por mes, con las que un escalón así es indetectable. El tramo
  2021-2025 **puede contener otros mantenimientos que no podemos ver**.
- `lr_fe_cu` perdió señal de Cu con el cambio (R² 0.206 → 0.092). Único caso
  donde el mantenimiento parece haber empeorado algo; revisar si se retoma Cu.

### 19.10 El "defecto de `dil`" NO hay que arreglarlo — corrige a §19.9

§19.9 anotó como defecto que `dil = logSumI − EWMA(90 d)` atraviesa la ventana
de recalibración de nov-2025 y durante ~3 meses absorbe un cambio de
instrumento como si fuera dilución. **El diagnóstico del mecanismo es
correcto** y se ve en la media mensual de `dil` (sd global 0.095):

```
2025-09 +0.017 | 2025-11 -0.009 | 2026-01 -0.079 | 2026-04 -0.043 | 2026-07 -0.147
2025-10 +0.016 | 2025-12 -0.075 | 2026-03 -0.012 | 2026-05 -0.114 | 2026-09 +0.034
```

Dic-ene es el escalón. Pero **may-jul es peor (−0.11 a −0.15)** y eso no es el
escalón: es la EWMA sin poder seguir la caída sostenida del `pSol` (2026Q1
20.55 → Q3 16.39).

**Medida la consecuencia, la conclusión se invierte.** Métrica del objetivo
(la de §10.7): efecto de `pSol` sobre `lr_fe_cu` a química fija, p10→p90 en sd
del feature. Sin corregir: **+0.896 sd**.

| variante | corr(dil,pSol) | b | efecto residual |
|---|---|---|---|
| **EWMA 90D global (actual)** | 0.622 | 0.1045 | **0.712** |
| EWMA 90D por época | 0.578 | 0.0877 | 0.752 |
| EWMA 90D, escalón re-nivelado | 0.595 | 0.0977 | 0.732 |
| EWMA 90D, re-nivelado + por época | 0.578 | 0.0877 | 0.752 |
| EWMA 30D / 14D / 7D global | 0.582 / 0.554 / 0.529 | — | 0.736 / 0.767 / 0.799 |
| EWMA 180D global | 0.634 | 0.1049 | 0.709 |
| EWMA 180D, escalón re-nivelado | 0.599 | 0.0990 | 0.729 |

**Las ocho variantes que quitan el escalón empeoran la corrección.** La única
que mejora es una memoria *más larga* (180D, 0.709 vs 0.712) y es diferencia en
la tercera cifra.

**El motivo, y es incómodo:** el escalón de `logSumI` fue **−7.4 %**, o sea en
la **misma dirección** en la que venía cayendo el `pSol`, y en los mismos
meses. Quitarlo le resta a `dil` algo que correlaciona con la dilución real.
**La corrección actual funciona en parte por coincidencia.**

- **Decisión: NO cambiar `dil`.** `DIL_HALFLIFE="90D"` y `DIL_CORREGIR=["lr_fe_cu"]`
  quedan como están.
- **Riesgo a registrar, con disparador concreto:** si un mantenimiento futuro
  empuja las intensidades en la dirección **contraria** a la del sólido, `dil`
  corromperá la corrección y con el signo equivocado. La salvaguarda no es
  cambiar la fórmula: es el **monitor de cocientes entre canales** (§19.2 da la
  firma) y **volver a ajustar `b` después de cada mantenimiento**.
- **El techo sigue donde estaba:** la corrección quita el 21 % del efecto de
  `pSol` (0.896 → 0.712) y nada de lo probado lo mejora. Subirlo exige `pSol` a
  cadencia de lectura, que no existe (§9.5).
- **Lección de método, la quinta de este tipo en el proyecto:** se diagnosticó
  el mecanismo y se dio por hecha la consecuencia sin medirla. El mecanismo era
  real y la consecuencia era la contraria.

Reproducible: `pixi run python scripts/variantes_senal_dilucion.py`.

### 19.11 Corrección: la campaña del Outocal no es un dataset aparte

§19.9 y el ADR del día listaban "conseguir los datos de la campaña del Outocal"
como lo de mayor valor pendiente, razonando que el muestreo de
`works4cdp_assay` no mostraba ningún pico tras noviembre (sigue plano en 5-13
por mes) y que por tanto la campaña estaría en otro sistema.

**Planta lo descartó: las muestras de esas fechas son las que ya están en la
base de datos.** No hay un conjunto denso aparte. El conjunto supervisado es el
que hay: 348 muestras con ley en todo el histórico, 64 en la ventana
post-mantenimiento. La ausencia de pico no era un dato faltante, era el dato.

### 19.12 El techo del Cu: su R² bajo es falta de varianza, no falla del modelo

Encargo: medir cuánto del error del Cu es irreducible. Dos métodos; el primero
no sirve y conviene que quede dicho.

**Variograma (descartado).** Extrapolar la semivarianza a lag 0 para estimar el
*nugget* exige conocer la forma por debajo de 24 h y **no hay datos ahí**: el
muestreo es cada 12 h y los pares de 12 h están contaminados por el turno (la
semivarianza a 24 h es MENOR que a 12 h). Los resultados salen absurdos: techo
0.571 para Mo cuando el modelo logra 0.783, y 1.000 para Cu en el test. El
"nugget" que estima es error de medición **más** variación de proceso más rápida
que 24 h, o sea una cota demasiado floja.

**Desacuerdo entre los dos cortes (sirve a medias).** Sobre 333 pares:

| ley | sd(dif entre cortes) | soporte | sd(resid modelo) | resid/dif | R² modelo |
|---|---|---|---|---|---|
| Cu | 2.223 | 0.743 | 1.460 | 0.66 | 0.452 |
| Fe | 1.602 | 0.327 | 1.141 | 0.71 | 0.119 |
| Mo | 0.863 | 0.199 | 0.367 | 0.43 | 0.783 |

El **soporte** —la variación real del material dentro de las 12 h, estimada con
la estimación por lectura— es chico: 0.743 de 2.223 en Cu. **El desacuerdo es
casi todo muestreo, no diferencia instante-vs-promedio.**

Suponer error igual en los dos cortes da un techo **por debajo** de lo que el
modelo logra, o sea el supuesto es falso: el **grab carga más error** que el
compósito por incrementos, que es lo que predice la teoría de muestreo. No se
puede repartir sin duplicados de ensayo, así que **el techo exacto no es
medible con lo que hay**. Lo que sí queda: **el residuo del modelo ya es ≤ el
desacuerdo entre las dos referencias.**

**Y la razón de fondo del R² bajo del Cu no es el error, es el denominador:**

```
ley    media     sd    CV %
cu     24.87   1.872    7.5
fe     28.19   1.301    4.6
mo      1.63   0.712   43.8
ins    11.08   2.595   23.4
```

El Cu del concentrado **casi no se mueve porque es el objetivo controlado**. El
0.783 del Mo no es porque el Mo se mida mejor: es que hay **6× más variación
relativa que explicar**. Mirando el cociente residuo/desacuerdo, el modelo no
es mucho peor con el Cu (0.66) que con el Mo (0.43).

**Decisión: no invertir en modelado de Cu.** Su error ya está en la resolución
de la referencia y hay poca varianza que explicar, poca **por diseño del
proceso**. Lo que desbloquearía la pregunta son **30-50 duplicados de ensayo del
compósito**: fijan el error de la referencia y el techo deja de ser incógnita.

### 19.13 Hay un sesgo de muestreo Cu↔ganga de −1.2 puntos entre los dos cortes

**Primero, una corrección de vocabulario que venía arrastrándose:** §11.3 y
§19.12 hablaban de "dos laboratorios". **Es el mismo laboratorio con el mismo
método** (confirmado con planta). Son dos **cortes** del mismo flujo:

```
flujo → MULTIPLEXOR ─corte→ balde 12 h          (COMPÓSITO; no pasa por el probe)
             │
             └──────→ PROBE (rayos X) ─corte→ recipiente  (CALIBRACIÓN)
```

El multiplexor está a **menos de 2 m** del probe y en ese tramo **no hay agua**;
el agua de dilución se aplica **muchos metros antes del multiplexor**, de modo
que **las dos muestras ven el mismo material ya diluido**. Los dos cortadores
toman la **sección completa** del flujo y el balde se **agita y se procesa
completo**. Y las muestras puntuales son las de **calibración del Outocal**:
intensidad y ensayo en la misma fila de la BD.

Eso corrige también a §12 y al `CLAUDE.md`, que decían que el compósito se corta
"antes del multiplexor": se corta **en** el multiplexor, antes del probe.

**El sesgo medido** (333 pares, 2023-01 → 2026-08):

| | diferencia | media ref. | relativo |
|---|---|---|---|
| Cu | **−1.200** | 25.71 | −4.7 % |
| Fe | −0.165 | 28.55 | −0.6 % |
| Mo | +0.115 | 1.38 | +8.4 % |
| Insoluble | **+1.160** | 9.41 | +12.3 % |

- **Es balance de masa Cu↔ganga:** `corr(dcu, dins) = −0.834`, y la suma
  `dcu+dins` da media −0.066 con sd 1.573 contra sd 2.252 y 2.845 de cada una.
  **Lo que el grab pierde en Cu lo gana en ganga, punto por punto.**
- **Es ADITIVO, no proporcional** (Bland-Altman, la diferencia contra el
  promedio: pendiente +0.113, p=0.093). Sesgo medio −1.200, IC95
  [−1.439, −0.961]. Eso importa: llevar el courier a la escala del flujo sería
  **una constante**, no una recalibración.
- **Límites de concordancia [−5.56, +3.16]:** una muestra puntual individual
  puede diferir del compósito de su ventana en 5 puntos de Cu. **Solo el
  promedio es comparable; par a par no.** Eso toca la etapa 9, que evalúa contra
  18 muestras de la BD en test — con esa dispersión el intervalo es muy ancho, y
  explica por qué el Kalman "empeora Cu" ahí de 0.52 a 0.24 sin que eso signifique
  que empeoró algo real.

**Lo que NO es, cada hipótesis con su medición:**

| descartado | con qué |
|---|---|
| diferencia analítica entre laboratorios | mismo laboratorio, mismo método |
| diferencia de soporte (instante vs 12 h) | solo 0.743 de los 2.223 de dispersión |
| agua de dilución | se aplica aguas arriba de los dos; `corr(dcu, pSol) = +0.006`, p=0.92 |
| la línea al probe | <2 m y sin agua |
| cortadores mal delimitados | toman sección completa, los dos |
| sedimentación en el balde | se agita y se procesa todo |
| hora de toma / ciclo diario | el ciclo tiene amplitud 0.839 y las 10-12 h están 0.179 bajo la ventana AM: explica el **15 %** |
| error de escala (proporcional) | Bland-Altman: aditivo |
| deriva en el tiempo | pendiente −0.205/año, p=0.080, no monótona (2023 −0.74, 2024 −1.22, 2025 −1.83, 2026 −0.84) |
| diferencia de BASE (húmeda/seca, normalización) | los relativos de los 4 elementos no son iguales ni del mismo signo (−4.7 %, −0.6 %, +8.4 %, +12.3 %); una base mueve todo por el mismo factor |

**El fenómeno queda sin explicación.** Es real, aditivo, constante y con firma
de intercambio de material — pero los cuatro mecanismos que se propusieron
cayeron con la descripción del muestreo. Falta información de proceso, no
análisis.

**El contraste que lo avanzaría sin tocar planta:** extraer los compósitos de
**cabeza** (`7100AIP011-014MAN`) y **relave** (`7100AIP031-034MAN`) y sus
muestras puntuales (`sample_id` 17 y 18 para cabeza), y ver si el mismo sesgo
relativo de ~−4.7 % en Cu aparece también ahí. Si está en las tres líneas es
común a la arquitectura de muestreo; si solo está en el concentrado, se busca en
esa línea. Es una extracción, no un ensayo de planta.

**Y lo que sí queda decidido**, que estaba abierto desde el ADR del 2026-09-30:
no son dos candidatas al mismo puesto. La **puntual es la referencia de
calibración por diseño** (ensaya el material que vio el probe, de ahí el calce
exacto de §12) y el **compósito es la auditoría independiente** (por
incrementos, representa el flujo). Cada una es correcta para su propósito, y el
−1.2 es el precio de que sean dos materiales distintos.

Reproducible: `pixi run python scripts/referencias_laboratorio.py`.

### 19.14 La estructura química del concentrado: dos perillas, y el Cu sin perilla propia

Medido el 2026-10-22 sobre la **química del laboratorio** (compósito de 12 h,
n = 2601, 2022-12-30 → 2026-09-29), sin instrumento en medio. Es la pregunta
previa a la arquitectura de estimación: qué se mueve con qué. Todo va además
**centrado por mes**, para que la tendencia lenta del insoluble de 2022 a 2026
no fabrique correlaciones.

| | Cu | Fe | Mo | ins |
|---|---|---|---|---|
| **CV** | 8.4 % | 5.2 % | **46.9 %** | 29.2 % |

| par | Pearson crudo | centrado por mes | parcial (controlando las otras dos) |
|---|---|---|---|
| Cu ~ Mo | +0.034 | −0.034 | −0.110 |
| Fe ~ Mo | −0.404 | −0.297 | −0.351 |
| Mo ~ ins | +0.176 | +0.060 | −0.097 |
| Cu ~ Fe | −0.206 | −0.030 | **−0.646** |
| Cu ~ ins | −0.566 | −0.714 | **−0.772** |
| Fe ~ ins | −0.474 | −0.394 | **−0.716** |

**Lo que dice, en tres frases:**

1. **El Mo es su propia perilla.** Se mueve mucho (CV 46.9 %) y **no arrastra
   al Cu** (corr ≈ 0 de las tres formas de medirla) **ni al insoluble**. Junto
   con §19.6 (gradiente, no grupos) explica por qué el Mo es el único resultado
   defendible: hay varianza que explicar y no está trabada con el resto.
2. **Pero sí arrastra al Fe**: −0.56 de Fe por cada punto de Mo (IC95 −0.62 a
   −0.50), igual centrado por mes (−0.559). **No es aritmética del cierre de
   masa**: si lo fuera, a Cu e insoluble fijos el coeficiente sería **−1**, y la
   suma de los cuatro sería constante; en cambio la suma (65.10 ± 1.70) **sube
   +0.52 por punto de Mo** (p=1.4e-28). Lo que no se mide (azufre, oxígeno)
   acompaña al Mo. Es química del mineral, no un artefacto de las fracciones.
   En varianza es modesto: el Mo explica el **9-16 %** de la del Fe (centrado /
   crudo).
3. **Cu, Fe y ganga son la otra perilla, trabada por balance de masa.** Cu y Fe
   casi no correlacionan entre sí en crudo (−0.03 centrado por mes), pero **a
   ganga fija se intercambian fuerte** (parcial −0.646), y los dos van contra el
   insoluble (parciales −0.77 y −0.72). Es la misma estructura que da
   `ins ~ cu + fe` con R² 0.627 (ADR 2026-10-21 (2)) y el intercambio Cu↔ganga
   de −1.2 puntos entre los dos cortes (§19.13). El **Cu no tiene perilla
   propia**: es el objetivo controlado y la planta lo sostiene (CV 8.4 %, §19.12).

**Lo que NO dice, y es el límite:** esto es la química. **Que el courier
resuelva el intercambio Cu↔Fe↔ganga no está demostrado**, y lo medido apunta a
que lo resuelve mal (R² fuera de fold Cu 0.545, Fe −0.170). La estructura fija
la **arquitectura** — modelos globales sobre variables continuas, sin grupos —,
no promete precisión en Cu y Fe.

**La pista para el modelo de Fe:** el Mo se estima bien (R² 0.78) y el Mo
explica parte del Fe con química real. Usar la estimación de Mo, o `lr_mo_cu`,
para el Fe es una apuesta medible, chica en tamaño esperado, y debe pasar el
walk-forward con control como todo lo demás.

Reproducible: `pixi run python scripts/estructura_quimica_composito.py`.

### 19.15 Dos decisiones de arquitectura: escala balde y Zn fuera de las leyes estimadas

**1. Las estimaciones se reportan en la escala del COMPÓSITO (el "balde"), en
las tres leyes.** Las dos fuentes difieren en Cu por −1.2 puntos aditivos y sin
explicación (§19.13); el modelo aprende de la puntual, así que habla en escala
"cucharada". Se elige la del compósito porque representa el flujo del turno,
que es lo que interesa a metalurgia, y porque, al no conocerse el origen del
−1.2, una constante fija podría quedar mal sin aviso: el Kalman lo sigue solo.
Consecuencias:

- el **Kalman es parte del modelo**, no una variante: sin él la salida está en
  escala cucharada;
- Fe (−0.18) y Mo (+0.07) van en la misma escala por consistencia — una sola
  regla;
- el **juez principal es la etapa 8** (compósito); la etapa 9 (puntual) pasa a
  ser **control de calibración**, donde lo esperado es ver la diferencia de
  ~1.2 en Cu. Ahí ya no se lee "el Kalman empeora el Cu";
- el Cap. 6 de la tesis declara la escala, el porqué y que el −1.2 sigue sin
  explicación.

**2. `pZn` sale de `TARGETS`; el canal `n3zn` se queda como feature.** Razones:
fuera del objetivo 1 (Cu, Fe, Mo); el compósito no reporta Zn, así que no tiene
auditoría, ni Kalman, ni puede ir en escala balde; y su R² de 0.898 en el
walk-forward descansa en la cola — mediana 0.151 %, CV 88.8 %, y las **15
muestras** del 5 % superior aportan el **80.2 %** de la suma de cuadrados: mide
"detecta los picos", no "estima el Zn diario". El test supervisado (18
muestras) puede no traer ningún pico. `n3zn_f` y `lr_zn_cu` (invariante a la
dilución por física, §19.3) **siguen entrando** a los modelos de Cu, Fe y Mo.
Un detector de picos de Zn queda como posible trabajo aparte, fuera de la tesis.

**Verificado que no mueve nada más:** etapas 5, 6, 8 y 9 corridas antes y
después del cambio — las 36 filas de métricas contra el compósito, las 9 contra
la BD y las predicciones fuera de fold de Fe, Cu y Mo son **idénticas**
(diferencia máxima 0.0). Los folds no dependen de `TARGETS` y el Zn era el
último en consumir el azar del control. La corrida de antes reproduce lo
registrado (Mo fuera de muestra sin anomalías, n=209: corr 0.913, R² 0.814 →
0.833 con Kalman) y queda como **línea base** del trabajo de modelado:

| ley | walk-forward R² | compósito corr | R² modelo → +Kalman | MAE modelo → +Kalman |
|---|---|---|---|---|
| Fe | −0.170 | 0.525 | −0.211 → 0.189 | 1.114 → 0.913 |
| Cu | 0.545 | 0.564 → 0.527 | 0.038 → 0.227 | 1.426 → 1.301 |
| Mo | 0.776 | 0.913 → 0.914 | 0.814 → 0.833 | 0.252 → 0.220 |

(compósito: fuera de muestra jun-sep 2026, sin anomalías, n=209). Revisa el ADR
del 2026-09-01 *«`dropna` por elemento»*, que conservaba el Zn para comparar
familias de regresores en V1; en V2, todo global, ese motivo ya no existe.
