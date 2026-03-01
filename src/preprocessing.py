import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop_duplicates()

    # Fill missing Postal Codes (if any)
    if 'Postal Code' in df.columns:
        df['Postal Code'] = df['Postal Code'].fillna(df['Postal Code'].mode()[0])

    # Remove extreme sales outliers (top 1%)
    if 'Sales' in df.columns:
        df = df[df['Sales'] < df['Sales'].quantile(0.99)]

    return df