# ============================================================
# src/database/extractor.py — Extracción de intensidades y ensayos desde Postgres
# ============================================================
# Idéntico en espíritu a src/database/data/getData.py de V1 (mismas queries,
# mismo esquema), reubicado a un solo nivel bajo src/database/ y con soporte
# opcional de filtro por rango de fechas (se aplica en pandas DESPUÉS de traer
# los datos, sin tocar las queries SQL originales que ya se sabe que funcionan).
# ============================================================

from src.database.connection import DBManager
from sqlalchemy import text  # Importante para usar params de forma segura


class Extractor(DBManager):
    def __init__(self, table_name: str, **kwargs):
        super().__init__(**kwargs)
        self.table_name = table_name

    def get_head(self):
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE lower(table_name) = lower(:t_name)
            AND table_schema = 'public'
            ORDER BY ordinal_position;
        """
        return self.execute_query(query, params={"t_name": self.table_name})

    def get_intensity(self, sample_id: int, desde: str | None = None, hasta: str | None = None):
        """Trae TODO el histórico de intensidades del sample_id (igual que V1).

        desde/hasta (opcionales, 'YYYY-MM-DD'): filtran por columna `date` DESPUÉS
        de traer los datos. Útil para pedir p.ej. "todo hasta agosto" sin tocar
        la query original.
        """
        query = f"""
            SELECT date, time, instance, n1fe, n2cu, n3zn, n4mo, n5ech5, n6sc, n7ech7
            FROM {self.table_name}
            WHERE sample_id = :s_id
            ORDER BY date ASC, time ASC
        """
        df = self.execute_query(query, params={"s_id": sample_id})
        return _filtrar_por_fecha(df, desde, hasta)

    def get_assays(self, sample_id: int, desde: str | None = None, hasta: str | None = None):
        query = f"""
            SELECT date, time, n1fe, n2cu, n3zn, n4mo, n5ech5, n6sc, n7ech7, "pFe", "pCu", "pZn", "pMo", "pIns", "pSol"
            FROM {self.table_name}
            WHERE sample_id = :s_id
            ORDER BY date ASC, time ASC
        """
        df = self.execute_query(query, params={"s_id": sample_id})
        return _filtrar_por_fecha(df, desde, hasta)


def _filtrar_por_fecha(df, desde, hasta):
    if desde is None and hasta is None:
        return df
    fechas = df["date"].astype(str)
    if desde is not None:
        df = df[fechas >= desde]
        fechas = df["date"].astype(str)
    if hasta is not None:
        df = df[fechas <= hasta]
    return df.reset_index(drop=True)
