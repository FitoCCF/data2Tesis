import pandas as pd
import os

def main():
    raw_dir = '/home/fito/Proyects/data2Tesis/data/raw'
    processed_dir = '/home/fito/Proyects/data2Tesis/data/processed'
    
    path_cobre = os.path.join(raw_dir, 'intensidades_cobre.csv')
    path_cobre_24 = os.path.join(raw_dir, 'intensidad_cobre_24.csv')
    path_output = os.path.join(processed_dir, 'intensidades_cobre.csv')
    
    print(f"Reading '{path_cobre}'...")
    df_cobre = pd.read_csv(path_cobre)
    print(f"Original shape: {df_cobre.shape}")
    print("Columns:", list(df_cobre.columns))
    
    print(f"\nReading '{path_cobre_24}'...")
    df_cobre_24 = pd.read_csv(path_cobre_24)
    print(f"Original shape: {df_cobre_24.shape}")
    print("Columns:", list(df_cobre_24.columns))
    
    # Map the columns of df_cobre to match df_cobre_24's names
    # df_cobre columns: ['date', 'time', 'cu', 'fe', 'mo', 'sc', 'zn']
    # df_cobre_24 columns: ['date', 'time', 'instance', 'n1fe', 'n2cu', 'n3zn', 'n4mo', 'n5ech5', 'n6sc', 'n7ech7']
    #
    # Mapping:
    # 'fe' -> 'n1fe'
    # 'cu' -> 'n2cu'
    # 'zn' -> 'n3zn'
    # 'mo' -> 'n4mo'
    # 'sc' -> 'n6sc'
    
    mapping = {
        'fe': 'n1fe',
        'cu': 'n2cu',
        'zn': 'n3zn',
        'mo': 'n4mo',
        'sc': 'n6sc'
    }
    
    df_cobre_mapped = df_cobre.rename(columns=mapping)
    
    # Reindex df_cobre_mapped to have the exact columns of df_cobre_24
    target_columns = list(df_cobre_24.columns)
    df_cobre_mapped = df_cobre_mapped.reindex(columns=target_columns)
    
    # Concatenate the datasets
    print("\nMerging datasets...")
    df_merged = pd.concat([df_cobre_mapped, df_cobre_24], ignore_index=True)
    print(f"Merged shape before sorting: {df_merged.shape}")
    
    # Sort by date and time to maintain temporal order
    df_merged['datetime'] = pd.to_datetime(df_merged['date'].astype(str) + ' ' + df_merged['time'].astype(str), errors='coerce')
    df_merged = df_merged.sort_values(by='datetime').drop(columns=['datetime'])
    
    print(f"Final merged shape: {df_merged.shape}")
    print(f"Saving merged dataset to '{path_output}'...")
    
    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    df_merged.to_csv(path_output, index=False)
    
    print("Verification of first 5 rows in merged file:")
    print(df_merged.head())
    print("Verification of last 5 rows in merged file:")
    print(df_merged.tail())
    print("\nProcess finished successfully!")

if __name__ == '__main__':
    main()
