import pandas as pd
import os

def parse_datetime_column(df):
    if 'date' in df.columns and 'time' in df.columns:
        return df

    if 'Unnamed: 0' in df.columns:
        dt = pd.to_datetime(df['Unnamed: 0'], errors='coerce')
        if dt.notna().any():
            df['date'] = dt.dt.strftime('%Y-%m-%d')
            df['time'] = dt.dt.strftime('%H:%M:%S')
            return df

    raise ValueError("No se pudo obtener date/time de df_cobre_24")

def main():
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    
    path_cobre = os.path.join(raw_dir, 'intensidad_cobre_db.csv')
    path_cobre_24 = os.path.join(raw_dir, 'Intensidades_20260717_1715.csv')
    path_output = os.path.join(processed_dir, 'intensidades_cobre.csv')
    path_train_output = os.path.join(processed_dir, 'intensidades_cobre_train.csv')
    path_test_output = os.path.join(processed_dir, 'intensidades_cobre_test.csv')
    
    print(f"Reading '{path_cobre}'...")
    df_cobre = pd.read_csv(path_cobre)
    print(f"Original shape: {df_cobre.shape}")
    print("Columns:", list(df_cobre.columns))
    
    print(f"\nReading '{path_cobre_24}'...")
    df_cobre_24 = pd.read_csv(path_cobre_24)
    print(f"Original shape: {df_cobre_24.shape}")
    print("Columns:", list(df_cobre_24.columns))
    
    df_cobre_24 = parse_datetime_column(df_cobre_24)

    mapping_24 = {
        'fe': 'n1fe',
        'cu': 'n2cu',
        'zn': 'n3zn',
        'mo': 'n4mo',
        'sc': 'n6sc'
    }
    df_cobre_24 = df_cobre_24.rename(columns=mapping_24)

    expected_columns = [
        'date', 'time', 'instance',
        'n1fe', 'n2cu', 'n3zn', 'n4mo', 'n5ech5', 'n6sc', 'n7ech7'
    ]

    df_cobre = df_cobre.reindex(columns=expected_columns)
    df_cobre_24 = df_cobre_24.reindex(columns=expected_columns)

    print("\nMerging datasets...")
    df_merged = pd.concat([df_cobre, df_cobre_24], ignore_index=True)

    df_merged['datetime'] = pd.to_datetime(
        df_merged['date'].astype(str) + ' ' + df_merged['time'].astype(str),
        errors='coerce'
    )
    df_merged = df_merged.sort_values(by='datetime').reset_index(drop=True)

    os.makedirs(processed_dir, exist_ok=True)
    df_merged.to_csv(path_output, index=False)

    split_index = int(len(df_merged) * 0.7)
    df_train = df_merged.iloc[:split_index].drop(columns=['datetime'])
    df_test = df_merged.iloc[split_index:].drop(columns=['datetime'])

    df_train.to_csv(path_train_output, index=False)
    df_test.to_csv(path_test_output, index=False)

    print("Saved:", path_output)
    print("Saved train split:", path_train_output)
    print("Saved test split:", path_test_output)

if __name__ == '__main__':
    main()