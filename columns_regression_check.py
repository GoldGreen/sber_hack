import pandas as pd

df = pd.read_parquet('data/processed_data.parquet', engine='fastparquet')

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]
