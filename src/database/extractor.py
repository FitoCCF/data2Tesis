# ============================================================
# src/database/extractor.py — Extracción desde Postgres (works4cdp_assay)
# ============================================================
# Una sola consulta, armada a partir de los grupos de columnas de
# config.COLS_BD. En la BD, la intensidad y la ley de laboratorio de una
# muestra están en la MISMA fila: pedirlas juntas evita tener que reunirlas
# después por 'instance'. Si una etapa necesita solo un grupo, lo pide con
# grupos=("leyes",) o selecciona columnas; no hace falta otra consulta.
# ============================================================

import pandas as pd
from sqlalchemy import text

from src.database.connection import crear_engine
from src.pipeline.config import TABLA_BD, LLAVES_BD, COLS_BD


class ExtractorBD:
    """Extractor de la tabla de ensayos. Abre una sola conexión (engine) y la
    reutiliza en todas las consultas de la instancia."""

    def __init__(self, db_config: dict | None = None, tabla: str = TABLA_BD):
        self.engine = crear_engine(db_config)
        self.tabla = tabla

    def extraer(self, sample_id: int, grupos=("intensidades", "leyes"),
                desde: str | None = None, hasta: str | None = None) -> pd.DataFrame:
        """Filas de un sample_id con las llaves + los grupos de columnas pedidos.

        grupos : nombres de config.COLS_BD ("intensidades", "leyes", ...).
        desde, hasta : 'YYYY-MM-DD', inclusivos, opcionales. Se filtran en SQL,
            no en pandas: la BD no devuelve el histórico completo para recortarlo.
        """
        desconocidos = [g for g in grupos if g not in COLS_BD]
        if desconocidos:
            raise ValueError(f"Grupos desconocidos {desconocidos}; disponibles: {list(COLS_BD)}")

        columnas = LLAVES_BD + [c for g in grupos for c in COLS_BD[g]]
        # Comillas dobles: Postgres respeta mayúsculas solo si el nombre va
        # entre comillas ("pFe"). Los nombres vienen de config, no del usuario.
        select = ", ".join(f'"{c}"' for c in columnas)

        condiciones = ["sample_id = :sample_id"]
        params = {"sample_id": sample_id}
        if desde is not None:
            condiciones.append("date >= :desde")
            params["desde"] = desde
        if hasta is not None:
            condiciones.append("date <= :hasta")
            params["hasta"] = hasta

        query = (f'SELECT {select} FROM "{self.tabla}" '
                 f'WHERE {" AND ".join(condiciones)} ORDER BY date, time')
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
