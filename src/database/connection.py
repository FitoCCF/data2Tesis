# src/database/connection.py — Configuración y conexión a Postgres
import os

from sqlalchemy import create_engine

# Config por defecto, sobreescribible por variables de entorno (DB_HOST,
# DB_PORT, DB_USER, DB_PASSWORD, DB_NAME): así el mismo código corre en
# desarrollo y en producción sin tocar nada. El puerto depende de la máquina:
# el contenedor de data4cdpv1/docker-compose.yml publica 5432; en la máquina
# donde se escribió V2 estaba publicado en 5433. Si no conecta, revisar
# `docker ps` y exportar DB_PORT.
DB_CONFIG_DEFAULT = {
    "user": os.environ.get("DB_USER", "myuser"),
    "password": os.environ.get("DB_PASSWORD", "mypassword"),
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": os.environ.get("DB_PORT", "5433"),
    "dbname": os.environ.get("DB_NAME", "mydb"),
}


def crear_engine(db_config: dict | None = None):
    """Engine de SQLAlchemy para la config dada (o la de por defecto)."""
    c = db_config or DB_CONFIG_DEFAULT
    return create_engine(f"postgresql://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['dbname']}")
