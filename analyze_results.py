from matplotlib import pyplot as plt
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

cat_boost_model =  model.estimators_[0]
xgb_model =  model.estimators_[1]
final_model = model.final_estimator_

coef = final_model.coef_[0]
inter = final_model.intercept_[0]

cat_boost_feature_importance = cat_boost_model.feature_importances_
xgb_feature_importance = xgb_model.feature_importances_ * 100

feature_df = pd.DataFrame({'Catboost' : cat_boost_feature_importance, 
                        'Xgb'  : xgb_feature_importance},
                        index=feature_cols)

feature_df = feature_df[(feature_df['Catboost'] >=1) | (feature_df['Xgb'] >= 1)]

with open("columns_last.txt", 'w') as f:
    f.writelines(col+'\n' for col in feature_df.index.tolist())

fig, ax = plt.subplots(figsize=(16,14))
plt.title(f'{inter:.1f} + {coef[0]:.1f} * Catboost + {coef[1]:.1f} * Xgb')
ax = feature_df.plot.bar(ax=ax)
ax.set_xlabel('Признак')
ax.set_ylabel('Важность, %')
plt.show()