import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

df = pd.read_parquet('data/untrashed_data.parquet', engine='fastparquet')

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('imputer', KNNImputer(n_neighbors=5)),
])

df[feature_cols] = pipeline.fit_transform(df[feature_cols])

# dim = PCA(n_components='mle')
# transformed = dim.fit_transform(df[feature_cols])

# converted_feature_columns = [f'feature_{i}' for i in range(dim.n_components_)]
# converted_df = pd.DataFrame(data=transformed, columns=converted_feature_columns, index = df.index)
# converted_df[target_col] = df[target_col]

# df = converted_df
# feature_cols = converted_feature_columns

print(df)

df.to_parquet('data/processed_data.parquet', engine='fastparquet')