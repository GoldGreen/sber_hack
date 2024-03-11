import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pickle

df = pd.read_parquet('data/untrashed_data.parquet', engine='fastparquet')

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('imputer', KNNImputer(n_neighbors=5)),
])

df[feature_cols] = pipeline.fit_transform(df[feature_cols].values)

with open("pipeline.pickle", "wb") as file:
    pickle.dump(pipeline, file)

df.to_parquet('data/processed_data.parquet', engine='fastparquet')

print(df)