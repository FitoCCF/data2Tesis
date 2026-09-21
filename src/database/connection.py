# src/database/connection.py
import os
from sqlalchemy import create_engine, text  # Asegúrate de importar text
import pandas as pd

# Config por defecto, sobreescribible por variables de entorno.
# NOTA: en V1 (data2Tesis) varios scripts (scripts/assay_int.py) tenían el
# puerto hardcodeado en 5432, pero el contenedor postgres_db real publica en
# 5433 (verificado con `docker ps`: 0.0.0.0:5433->5432/tcp). Se corrige aquí
# y se centraliza para que ningún script vuelva a hardcodearlo mal.
DB_CONFIG_DEFAULT = {
    "user": os.environ.get("DB_USER", "myuser"),
    "password": os.environ.get("DB_PASSWORD", "mypassword"),
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": os.environ.get("DB_PORT", "5433"),
    "dbname": os.environ.get("DB_NAME", "mydb"),
}


class DBManager:
    def __init__(self, user, password, host, port, dbname):
        self.url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(self.url)

    def execute_query(self, query, params=None):
        with self.engine.connect() as conn:
            # Crucial: envolvemos 'query' en text()
            return pd.read_sql(text(query), conn, params=params)
