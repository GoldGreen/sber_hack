import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

df = pd.read_parquet('data/train_ai_comp_final_dp.parquet', engine='fastparquet').set_index('id').drop(['feature756', 'sample_ml_new'], axis=1)


target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]

imputer = SimpleImputer(strategy='mean')
df[feature_cols] = imputer.fit_transform(df[feature_cols])

scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# vif_data = pd.DataFrame()
# vif_data["feature"] = df.columns
# vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

# correlation_matrix = df.corr().abs()

# mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)

# correlated_features = set()

# for i in range(len(correlation_matrix.columns)):
#     print(f'check column {df.columns[i]}')
#     for j in range(i):
#         if mask[i, j] and correlation_matrix.iloc[i, j] >= 0.8:
#             colname = correlation_matrix.columns[i]
#             correlated_features.add(colname)

# print(correlated_features)
# df = df.drop(list(correlated_features), axis=1)

df.to_parquet('data/processed_data.parquet', engine='fastparquet')