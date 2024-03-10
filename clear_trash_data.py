import numpy as np
import pandas as pd
from scipy import stats
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

df = pd.read_parquet('data/train_ai_comp_final_dp.parquet', engine='fastparquet').set_index('id').drop(['feature756', 'sample_ml_new'], axis=1)
original_columns = df.columns.tolist()

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]

#sampler = RandomOverSampler()
sampler = RandomUnderSampler()

x_resampled, y_resampled = sampler.fit_resample(df[feature_cols], df[target_col])

resampled_df = pd.concat([x_resampled, y_resampled], axis=1)

def removeColumns(df:pd.DataFrame, feature_cols:list[str], cols:list[str]) -> pd.DataFrame:
    for col in cols:
        feature_cols.remove(col)

    return df.drop(cols, axis=1)

def check_pearson(feature:pd.Series,  target:pd.Series, threshold:float):
    not_null_series_idx = ~feature.isnull()

    return abs(stats.pearsonr(feature[not_null_series_idx], target[not_null_series_idx])[0]) < threshold


resampled_df = removeColumns(resampled_df, feature_cols, [col for col in feature_cols if resampled_df[col].nunique() == 1])
resampled_df = removeColumns(resampled_df, feature_cols, [col for col in feature_cols if check_pearson(resampled_df[col], resampled_df[target_col], 0.15)])
                   
corr = resampled_df.corr()
upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
resampled_df = removeColumns(resampled_df, feature_cols, [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.7)])

df = df.drop(df.columns.difference(resampled_df.columns).tolist(), axis=1)

print(df)

df.to_parquet('data/untrashed_data.parquet', engine='fastparquet')
