# src/database/connection.py
from sqlalchemy import create_engine, text # Asegúrate de importar text
import pandas as pd

class DBManager:
    def __init__(self, user, password, host, port, dbname):
        self.url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(self.url)

    def execute_query(self, query, params=None):
        with self.engine.connect() as conn:
            # Crucial: envolvemos 'query' en text()
            return pd.read_sql(text(query), conn, params=params)
