# Estimación en línea desde la máquina de oficina

Guía para dejar una máquina estimando Cu, Fe y Mo del Concentrado Colectivo
toda la noche y comparar al día siguiente contra el compósito de laboratorio.
Escrita el 2026-10-24, tras actualizar BD + PI y correr el pipeline completo.

> **La IP de la pasarela PI NO se escribe en ningún archivo** (cambia seguido).
> Se pasa solo por variable de entorno. Donde dice `<IP_PASARELA>`, poner la
> vigente. La pasarela de esta instalación escucha en el puerto **5173** y
> **no pide token** (`PI_TOKEN` vacío).

---

## 0. Requisitos

- `git` y [`pixi`](https://pixi.sh) instalados.
- Acceso de red a `<IP_PASARELA>:5173` (probar con `curl http://<IP_PASARELA>:5173`
  o desde el navegador).
- La BD Postgres **no hace falta**: `data/raw/courier_bd.csv` viene en el repo
  (se versiona con `git add -f`, igual que los demás crudos y el bundle de
  `models/`).

## 1. Traer el código y los datos

Primera vez:

```bash
git clone -b data2TesisV2 git@github.com:FitoCCF/data2Tesis.git data2TesisV2
cd data2TesisV2
pixi install
```

Si ya está clonado:

```bash
cd data2TesisV2
git checkout data2TesisV2
git pull
pixi install
```

Comprobar que llegaron los crudos (deben existir y no estar vacíos):

```bash
ls -la data/raw/courier_bd.csv data/raw/courier_pi.csv data/raw/composito_pi.csv
```

## 2. Correr el pipeline completo

### 2a. Extraer lo nuevo del PI (courier y compósito)

Sin la BD, se extraen solo las dos fuentes del PI; `courier_bd.csv` queda el
del repo.

```bash
PI_GATEWAY_HOST=<IP_PASARELA> PI_GATEWAY_PORT=5173 PI_TOKEN= \
  pixi run python notebooks/00_getdata.py --fuentes pi composito
```

Si la máquina **sí** ve la BD (contenedor `data4cdpv1-postgres_db-1`), puede
extraerse también: agregar `bd` a `--fuentes` y `DB_PORT=5432` (el default del
código es 5433 y hoy el contenedor publica en 5432).

Salida esperada: `courier_pi.csv` con ~436 mil filas desde 2025-07-13 y
`composito_pi.csv` con ~2650 ensayos desde 2022-12-30.

### 2b. Etapas 1a → 9

```bash
for e in 01a_limpieza_fuentes 01b_unificacion 01c_limpieza 02_features \
         03_escalado 04_clusterizado 05_modelos_locales 06_inferencia_hibrida \
         08_evaluacion_laboratorio 09_evaluacion_vs_lab_real; do
  echo "== $e"
  MPLBACKEND=Agg pixi run python notebooks/$e.py > /dev/null || { echo "FALLÓ $e"; break; }
done
```

En Windows (PowerShell):

```powershell
$env:MPLBACKEND = "Agg"
foreach ($e in "01a_limpieza_fuentes","01b_unificacion","01c_limpieza","02_features",
               "03_escalado","04_clusterizado","05_modelos_locales","06_inferencia_hibrida",
               "08_evaluacion_laboratorio","09_evaluacion_vs_lab_real") {
  Write-Host "== $e"; pixi run python "notebooks/$e.py" | Out-Null
  if ($LASTEXITCODE -ne 0) { Write-Host "FALLO $e"; break }
}
```

### 2c. Verificar que reproduce la corrida del 2026-10-24

`data/final/eval_composito_metricas.csv`, filas `sin anomalías` /
`fuera de muestra`, `modelo + Kalman` (n = 253):

| Ley | R² | MAE |
|---|---|---|
| pMo | 0.839 | 0.205 |
| pCu | 0.265 | 1.230 |
| pFe | 0.160 | 0.887 |

`data/final/eval_bd_metricas.csv`, `test (BD)`, n = 25: Mo + Kalman R² 0.723.

Si el n del compósito sale mayor que 253 es normal: el laboratorio publicó
compósitos nuevos con lecturas del courier. Si las cifras de Mo se alejan
mucho, **no** seguir al paso 3 y revisar antes.

## 3. Dejar el modelo en línea toda la noche

```bash
PI_GATEWAY_HOST=<IP_PASARELA> PI_GATEWAY_PORT=5173 PI_TOKEN= MPLBACKEND=Agg \
  nohup pixi run python -u scripts/en_linea.py --cada 20 > en_linea.log 2>&1 &
```

PowerShell (dejar la ventana abierta):

```powershell
$env:PI_GATEWAY_HOST = "<IP_PASARELA>"; $env:PI_GATEWAY_PORT = "5173"; $env:PI_TOKEN = ""
$env:MPLBACKEND = "Agg"
pixi run python -u scripts/en_linea.py --cada 20 *> en_linea.log
```

Cada 20 min: extrae lo nuevo del PI (incremental, 2 h de solape), corre 1a → 6
con el bundle **congelado** (no reentrena) y estima cada compósito pendiente
(07:30 y 19:30, ventana centrada de 12 h, escala balde con Kalman).

- **Desactivar la suspensión** de la máquina: si se duerme, se pierden ciclos.
  Los datos no se pierden (el siguiente ciclo extrae desde el último dato),
  pero conviene que corra continuo.
- Seguimiento: `tail -f en_linea.log` (o `Get-Content en_linea.log -Wait`).
- Un ciclo fallido imprime `CICLO FALLÓ: ...` y el monitor sigue; lo típico es
  la pasarela caída o con otra IP.

## 4. Mañana: comparar contra el compósito

El laboratorio publica el compósito con ~20 h de retraso. Cuando aparece en el
PI, el siguiente ciclo lo compara solo. La tabla del log muestra, por
compósito, `estimado vs laboratorio (diferencia)` para Cu, Fe y Mo, con estado
`lab` (comparado), `ventana cerrada` (esperando al laboratorio) o `en curso`.

Lo mismo queda en `data/final/en_linea.csv` (una fila por compósito desde
2026-09-01): `<ley>_est` (modelo), `<ley>_kal` (escala balde, la que se
reporta), `p<ley>` (laboratorio), `n_bloques` (de 25; con menos de 8 no hay
estimación) y `publicado`.

Compósitos de esta noche:

| Compósito | Ventana de lecturas |
|---|---|
| 2026-10-24 19:30 | 24-oct 13:30 → 25-oct 01:30 |
| 2026-10-25 07:30 | 25-oct 01:30 → 25-oct 13:30 |

Referencia para leer las diferencias: en producción (1-sep → 21-oct, n=100) el
Mo dio corr 0.925 / R² 0.852 y el Cu MAE ~0.99. **No juzgar Cu ni Fe por un
compósito suelto**: entre las dos referencias de laboratorio los límites de
concordancia del Cu son [−5.56, +3.16].

## 5. Qué NO hacer

- No escribir la IP en `config.py` ni en ningún archivo versionado.
- No comitear desde la oficina los CSV que el monitor va actualizando sin
  revisarlo antes: mueven la línea base.
- No reentrenar (etapa 5) con los datos de la noche: la comparación de mañana
  tiene que ser con el modelo congelado.
