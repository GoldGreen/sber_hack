from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
plt.style.use('ggplot') 

df = pd.read_parquet('data/pred_untrashed_data.parquet', engine='fastparquet')

target_col = 'target'
pred_col = 'target'

feature_cols = [column for column in df.columns if column not in [target_col] ]

targets = df[target_col].unique()

for col in df.columns:
    if col != target_col and col != pred_col:
        plt.figure()
        plt.hist([df.loc[df[target_col] == target, col] for target in targets], label=targets, density=True, stacked=True)
        
        plt.title(col)
        plt.xlabel('value')
        plt.ylabel('count')
        plt.legend()
        plt.show()