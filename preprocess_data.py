import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
 
df = pd.read_parquet('data/train_ai_comp_final_dp.parquet', engine='fastparquet').set_index('id').drop(['feature756', 'sample_ml_new'], axis=1)

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]

df = pd.concat([df[df[target_col] == 1], df[df[target_col] == 0].sample(n=(df[target_col] == 1).sum())])

imputer = SimpleImputer(strategy='median')
df[feature_cols] = imputer.fit_transform(df[feature_cols])

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

dim = PCA(n_components=103)
transformed = dim.fit_transform(df[feature_cols])

converted_feature_columns = [f'feature_{i}' for i in range(dim.n_components_)]
converted_df = pd.DataFrame(data=transformed, columns=converted_feature_columns, index = df.index)
converted_df[target_col] = df[target_col]

df = converted_df
feature_cols = converted_feature_columns

df.to_parquet('data/processed_data.parquet', engine='fastparquet')