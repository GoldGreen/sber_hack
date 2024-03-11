import turtle
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score,accuracy_score
from imblearn.under_sampling import RandomUnderSampler

from sklearn.pipeline import Pipeline

with open("pipeline.pickle", "rb") as file:
    pipeline: Pipeline = pickle.load(file)

with open("model.pickle", "rb") as file:
    model_dic = pickle.load(file)

cluster_model: KMeans = model_dic['cluster_model']
model: CatBoostClassifier = model_dic['model']

with open("columns.txt") as file:
    feature_cols = [line.rstrip() for line in file]
target_col = 'target'


def predict(array: np.ndarray):
    array = pipeline.transform(array)
    cluster_values_array = cluster_model.predict(array)
    cluster_dummies_array = np.eye(len(np.unique(cluster_values_array)))[cluster_values_array]
    array = np.concatenate((array, cluster_dummies_array), axis=1)

    return model.predict(array)


df = pd.read_parquet('data/train_ai_comp_final_dp.parquet', engine='fastparquet')

x: np.ndarray
y: np.ndarray
x, y = RandomUnderSampler().fit_resample(df[feature_cols].values, df[target_col].values)

y_pred = predict(x)

roc_auc = roc_auc_score(y, y_pred)
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
matrix = pd.DataFrame(confusion_matrix(y, y_pred), index=['TN', 'TP'], columns=['TN', 'TP'])

print(matrix)
print(f"Roc_auc Score: {roc_auc:.3f}")
print(f"Accuracy Score: {accuracy:.3f}")
print(f"F1 Score: {f1:.3f}")