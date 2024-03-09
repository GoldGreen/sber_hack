import pandas as pd

df = pd.read_parquet('data/train_ai_comp_final_dp.parquet', engine='fastparquet')
print(df)