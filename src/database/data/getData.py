from src.database.connection import DBManager
from sqlalchemy import text # Importante para usar params de forma segura

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

    def get_intensity(self, sample_id: int):
        # Query optimizada para el sensor PSI
        query = f"""
            SELECT date, time, instance, n1fe, n2cu, n3zn, n4mo, n5ech5, n6sc, n7ech7 
            FROM {self.table_name} 
            WHERE sample_id = :s_id
        """
        return self.execute_query(query, params={"s_id": sample_id})
    

