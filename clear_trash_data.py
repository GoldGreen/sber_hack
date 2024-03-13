from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from scipy import stats
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

df = pd.read_parquet('data/train_ai_comp_final_dp.parquet', engine='fastparquet').set_index('id').drop(['feature756', 'sample_ml_new'], axis=1)
original_columns = df.columns.tolist()

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]

model = CatBoostClassifier(auto_class_weights = 'Balanced',  
                           learning_rate = 0.08,    
                           iterations = 200)

model.fit(df[feature_cols], df[target_col])

feature_importance = model.feature_importances_
important_columns = [col for col, importance in zip(feature_cols, feature_importance) if importance >=  0.1]

removed_columns = df[feature_cols].columns.difference(important_columns).tolist()

df = df.drop(removed_columns, axis=1)
feature_cols = [column for column in df.columns if column not in [target_col] ]

with open("columns.txt", 'w') as f:
    f.writelines(col+'\n' for col in feature_cols)

df.to_parquet('data/untrashed_data.parquet', engine='fastparquet')

print(df)
