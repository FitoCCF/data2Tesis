import pandas as pd
import os

def main():
    file_path = '/home/fito/Proyects/data2Tesis/data/raw/intensidades_cobre.csv'
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    
    print("\nOriginal columns:", list(df.columns))
    print("Original first 5 rows:")
    print(df.head())
    
    # Split 'date' column by the space
    # The format is 'YYYY-MM-DD HH:MM:SS'
    # So split(' ', n=1, expand=True) will give date in [0] and time in [1]
    split_df = df['date'].astype(str).str.split(' ', n=1, expand=True)
    
    # Insert 'time' column right after 'date' (which is at index 0)
    df.insert(1, 'time', split_df[1])
    df['date'] = split_df[0]
    
    print("\nModified columns:", list(df.columns))
    print("Modified first 5 rows:")
    print(df.head())
    
    # Save the modified dataframe back to the same file
    df.to_csv(file_path, index=False)
    print("\nFile saved successfully.")

if __name__ == '__main__':
    main()
