import pandas as pd
import numpy as np

def load_stock_data(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df

def resample_to_frequency(df_1min, freq='15min'):
    df_resampled = df_1min.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return df_resampled

def align_data(df1, df2):
    common_index = df1.index.intersection(df2.index)
    
    return df1.loc[common_index], df2.loc[common_index]

def get_price_series(df1, df2, price_column='close'):
    series1 = df1[price_column]
    series2 = df2[price_column]
    
    mask = ~(series1.isna() | series2.isna())
    
    return series1[mask], series2[mask]