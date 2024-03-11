from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_parquet('data/untrashed_data.parquet', engine='fastparquet')

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]

threshold_unique_values = 10
categorical_features = [col for col in feature_cols if df[col].nunique() <= threshold_unique_values]

df[categorical_features] = df[categorical_features].astype('category')

data = {
    'Категориальная переменная': ['A', 'B', 'A', 'C', 'B', 'C'],
    'Числовая переменная': [10, 20, 30, 15, 25, 35]
}

numerical_columns = df[feature_cols].select_dtypes(include=['int', 'float']).columns
categorical_columns = df[feature_cols].select_dtypes(include=['object']).columns

if False:
    colors = ['green','red']
    for column in numerical_columns:
        plt.figure(figsize=(8, 6))
        for target_value, color in zip(df['target'].unique(), colors):
            data = df.loc[df['target'] == target_value, column]
            plt.hist(data, color=color, alpha=0.7, bins=5, label=f'Target {target_value}')
        plt.title(f'Распределение {column}')
        plt.xlabel(column)
        plt.ylabel('Количество')
        plt.legend()
        plt.show()

    for column in categorical_columns:
        plt.figure(figsize=(8, 6))
        for target_value, color in zip(df['target'].unique(), colors):
            data = df.loc[df['target'] == target_value, column]
            data.value_counts().plot(kind='bar', color=color, alpha=0.7, label=f'Target {target_value}')
        plt.title(f'Распределение {column}')
        plt.xlabel(column)
        plt.ylabel('Количество')
        plt.legend()
        plt.show()

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('imputer', KNNImputer(n_neighbors=5)),
])

df[feature_cols] = pipeline.fit_transform(df[feature_cols].values)

with open("pipeline.pickle", "wb") as file:
    pickle.dump(pipeline, file)

print(df)

df.to_parquet('data/processed_data.parquet', engine='fastparquet')