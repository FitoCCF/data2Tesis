#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# SCRIPT: assay_int.py
# DESCRIPCIÓN: Procesa y fusiona los datos de ensayos químicos ('assays') de la
#              base de datos con el dataset de intensidades clusterizado,
#              generando un conjunto de datos completo y otro filtrado.
# ==============================================================================

import os # Biblioteca estándar para interactuar con el sistema operativo (directorios, rutas)
import sys # Biblioteca estándar para interactuar con variables y funciones del sistema/runtime

# ------------------------------------------------------------------------------
# Configuración del entorno y directorios
# ------------------------------------------------------------------------------

# Obtener la ruta absoluta del directorio donde se encuentra este script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Obtener la ruta del directorio raíz del proyecto (un nivel arriba de scripts/)
project_root = os.path.dirname(script_dir)

# Añadir el directorio raíz del proyecto al path del sistema para poder importar 'src'
sys.path.append(project_root)

# Importar pandas para el análisis y manipulación de datos estructurados
import pandas as pd

# Importar la clase Extractor desde el paquete src.database para extraer datos
from src import Extractor

# ------------------------------------------------------------------------------
# Parámetros y configuraciones del proceso
# ------------------------------------------------------------------------------

# Nombre de la tabla en la base de datos de donde se extraerán los datos
TABLE_NAME = "works4cdp_assay"

# ID de la muestra o sensor a analizar (específico para este proceso)
SAMPLE_ID = 24

# Diccionario con la configuración detallada para la conexión a la base de datos PostgreSQL
db_config = {
    'user': 'myuser',      # Nombre del usuario de la base de datos
    'password': 'mypassword',  # Contraseña de acceso a la base de datos
    'host': 'localhost',   # Dirección del servidor de la base de datos
    'port': '5432',        # Puerto de escucha del servicio de base de datos
    'dbname': 'mydb'       # Nombre de la base de datos a conectar
}

def clean_time_string(time_value):
    """
    Función explícita para limpiar y normalizar el valor de la hora.
    Convierte a string, elimina espacios y quita la fracción de segundos si existe.
    """
    # Convertir el valor recibido a una cadena de texto (string)
    time_str = str(time_value)
    # Eliminar cualquier espacio en blanco al inicio o al final de la cadena
    time_cleaned = time_str.strip()
    # Si la cadena contiene un punto decimal (fracción de segundos, ej: 14:30:15.123)
    if '.' in time_cleaned:
        # Dividir la cadena por el punto y conservar únicamente la parte entera (antes del punto)
        time_cleaned = time_cleaned.split('.')[0]
    # Retornar el valor de la hora limpio y formateado
    return time_cleaned

def main():
    # --------------------------------------------------------------------------
    # Inicialización del Extractor y descarga de datos
    # --------------------------------------------------------------------------
    
    # Imprimir mensaje informativo del inicio del proceso
    print("Inicializando conexión con la base de datos...")
    # Crear una instancia del objeto Extractor con la tabla y los parámetros de conexión
    extractor = Extractor(table_name=TABLE_NAME, **db_config)
    
    # Imprimir mensaje indicando la descarga de los datos
    print(f"Obteniendo datos de ensayos para la muestra (sample_id): {SAMPLE_ID}...")
    # Ejecutar la consulta en la base de datos para obtener los ensayos (assays)
    df_assays = extractor.get_assays(SAMPLE_ID)
    
    # --------------------------------------------------------------------------
    # Lectura del dataset clusterizado
    # --------------------------------------------------------------------------
    
    # Definir la ruta relativa/absoluta al archivo CSV clusterizado en data/processed
    cluster_csv_path = os.path.join(project_root, 'data', 'processed', 'intensidad_cobre_24_clusterizado.csv')
    # Imprimir la ruta del archivo que se va a leer
    print(f"Cargando dataset clusterizado desde: {cluster_csv_path}...")
    
    # Leer el archivo CSV usando pandas
    df_cluster_raw = pd.read_csv(cluster_csv_path)
    # Eliminar aquellas filas del dataset clusterizado que no tengan un valor válido en la columna 'instance'
    df_cluster = df_cluster_raw.dropna(subset=['instance'])
    
    # --------------------------------------------------------------------------
    # Limpieza de fechas y horas en los ensayos
    # --------------------------------------------------------------------------
    
    # Imprimir mensaje del paso de limpieza en df_assays
    print("Normalizando columnas 'date' y 'time' en df_assays...")
    # Convertir todos los valores de la columna 'date' a strings y quitar espacios sobrantes
    df_assays['date'] = df_assays['date'].astype(str).str.strip()
    # Aplicar la función explícita clean_time_string a cada celda de la columna 'time'
    df_assays['time'] = df_assays['time'].apply(clean_time_string)
    
    # --------------------------------------------------------------------------
    # Selección y limpieza de columnas del clustering
    # --------------------------------------------------------------------------
    
    # Definir de forma explícita las columnas que necesitamos del dataset de clustering
    cols_to_select = [
        'date',            # Columna de fecha para la fusión (key)
        'time',            # Columna de hora para la fusión (key)
        'instance',        # Columna que identifica la instancia o punto de medición
        'n1fe_ortho',      # Medición normalizada u ortogonal de Hierro (Fe)
        'n2cu_ortho',      # Medición normalizada u ortogonal de Cobre (Cu)
        'n3zn_ortho',      # Medición normalizada u ortogonal de Zinc (Zn)
        'n4mo_ortho',      # Medición normalizada u ortogonal de Molibdeno (Mo)
        'cluster_kmeans',  # Asignación de cluster usando K-Means
        'cluster_gmm'      # Asignación de cluster usando Gaussian Mixture Models
    ]
    
    # Crear un subconjunto copia del dataframe clusterizado únicamente con las columnas seleccionadas
    df_cluster_subset = df_cluster[cols_to_select].copy()
    
    # Imprimir mensaje de limpieza en el subconjunto de clustering
    print("Normalizando columnas 'date' y 'time' en df_cluster_subset...")
    # Convertir los valores de 'date' en df_cluster_subset a string y limpiar espacios
    df_cluster_subset['date'] = df_cluster_subset['date'].astype(str).str.strip()
    # Aplicar la función explícita clean_time_string a la columna 'time' en df_cluster_subset
    df_cluster_subset['time'] = df_cluster_subset['time'].apply(clean_time_string)
    
    # --------------------------------------------------------------------------
    # Fusión de datasets (Merge)
    # --------------------------------------------------------------------------
    
    # Imprimir mensaje de fusión
    print("Fusionando df_assays con df_cluster_subset (Inner Join)...")
    # Realizar una fusión de tipo 'inner join' usando las columnas 'date' y 'time' comunes a ambos
    df_merged = pd.merge(df_assays, df_cluster_subset, on=['date', 'time'], how='inner')
    
    # Definir la ruta de salida para guardar el archivo completo fusionado
    output_complete_path = os.path.join(project_root, 'data', 'processed', 'intensidad_cobre_24_completo.csv')
    # Imprimir la ruta donde se guardará
    print(f"Guardando datos completos en: {output_complete_path}...")
    # Guardar el dataframe fusionado a CSV sin incluir el índice automático de pandas
    df_merged.to_csv(output_complete_path, index=False)
    
    # --------------------------------------------------------------------------
    # Filtrado y guardado del dataset filtrado
    # --------------------------------------------------------------------------
    
    # Columnas específicas utilizadas para filtrar filas vacías (deben tener al menos un valor no nulo)
    filter_columns = [
        'pFe',  # Porcentaje/valor de Hierro (Fe) de laboratorio
        'pCu',  # Porcentaje/valor de Cobre (Cu) de laboratorio
        'pZn',  # Porcentaje/valor de Zinc (Zn) de laboratorio
        'pMo',  # Porcentaje/valor de Molibdeno (Mo) de laboratorio
        'pIns', # Insoluble
        'pSol'  # Soluble
    ]
    
    # Filtrar el dataframe eliminando las filas donde todos los valores en 'filter_columns' sean nulos (dropna con how='all')
    df_final = df_merged.dropna(subset=filter_columns, how='all')
    
    # Definir la ruta de salida para el archivo filtrado
    output_filtered_path = os.path.join(project_root, 'data', 'processed', 'intensidad_cobre_24_completo_filtrado.csv')
    # Imprimir la ruta del archivo filtrado
    print(f"Guardando datos filtrados en: {output_filtered_path}...")
    # Guardar el dataframe filtrado a CSV sin incluir el índice automático de pandas
    df_final.to_csv(output_filtered_path, index=False)
    
    # Imprimir mensaje de éxito indicando que el proceso finalizó correctamente
    print("Proceso completado exitosamente.")

if __name__ == '__main__':
    # Llamar a la función principal del script si se ejecuta de manera directa
    main()
