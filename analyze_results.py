import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import StackingClassifier

with open("model.pickle", "rb") as file:
    model_dic = pickle.load(file)

model: StackingClassifier = model_dic['model']

with open("columns.txt") as file:
    feature_cols = [line.rstrip() for line in file]
target_col = 'target'


def predict(array: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    prob = model.predict_proba(array)
    prob = prob[:, 1]

    return (prob >= 0.5).astype(int), prob

df = pd.read_parquet('data/test_sber.parquet', engine='fastparquet').set_index('id')

x: np.ndarray
y: np.ndarray
x = df[feature_cols].values

importance = np.array([estimator.feature_importances_ for estimator  in model.estimators_]).sum()
print(importance)