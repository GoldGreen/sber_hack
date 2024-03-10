import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from scipy import stats
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

df = pd.read_parquet('data/train_ai_comp_final_dp.parquet', engine='fastparquet').set_index('id').drop(['feature756', 'sample_ml_new'], axis=1)

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]

#oversampler = RandomOverSampler()
oversampler = RandomUnderSampler()

x_resampled, y_resampled = oversampler.fit_resample(df[feature_cols], df[target_col])

df = pd.concat([x_resampled, y_resampled], axis=1)


def removeColumns(df:pd.DataFrame, feature_cols:list[str], cols:list[str]):
    for col in cols:
        feature_cols.remove(col)

    return df.drop(cols, axis=1)

def check_pearson(feature:pd.Series,  target:pd.Series, threshold):
    not_null_series_idx = ~feature.isnull()

    return abs(stats.pearsonr(feature[not_null_series_idx], target[not_null_series_idx])[0]) < threshold


a = df[target_col]

df = removeColumns(df, feature_cols, [col for col in feature_cols if df[col].nunique() == 1])
df = removeColumns(df, feature_cols, [col for col in feature_cols if check_pearson(df[col], df[target_col], 0.15)])
                   
corr = df.corr()
upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
df = removeColumns(df, feature_cols, [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.7)])


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