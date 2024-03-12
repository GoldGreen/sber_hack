from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score,accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.pipeline import Pipeline

with open("model.pickle", "rb") as file:
    model_dic = pickle.load(file)

cluster_model: KMeans = model_dic['cluster_model']
model: CatBoostClassifier = model_dic['model']
inputer: KNNImputer = model_dic['inputer']

with open("columns.txt") as file:
    feature_cols = [line.rstrip() for line in file]
target_col = 'target'


def predict(array: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    # array = inputer.transform(array)
    # cluster_values_array = cluster_model.predict(array)
    # cluster_dummies_array = np.eye(cluster_model.n_clusters)[cluster_values_array]
    # array = np.concatenate((array, cluster_dummies_array), axis=1)

    prob = model.predict(array, prediction_type='Probability')
    prob = prob[:, 1]

    return (prob >= 0.5).astype(int), prob


df = pd.read_parquet('data/test_sber.parquet', engine='fastparquet').set_index('id')

x: np.ndarray
y: np.ndarray
x = df[feature_cols].values

y_pred, y_prob = predict(x)

res_df = pd.DataFrame(data=np.column_stack((y_pred, y_prob)), index=df.index, columns=['target_bin', 'target_prob'])
res_df['target_bin'] = res_df['target_bin'].astype(int)
res_df.to_csv('data/result.csv')