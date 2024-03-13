import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgbm

from sklearn.utils import class_weight

df = pd.read_parquet('data/untrashed_data.parquet', engine='fastparquet')

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]

x_train, y_train = df[feature_cols], df[target_col]


cat_boost_model = CatBoostClassifier(verbose=False,
                           depth = 6,
                           l2_leaf_reg = 7,
                           learning_rate = 0.048,
                           iterations=700,
                           boosting_type='Plain',
                           bootstrap_type='Bernoulli',
                           subsample= 0.88,
                           rsm=0.88,
                           random_strength = 0.7,
                           leaf_estimation_iterations=5,
                           max_ctr_complexity = 1,
                           leaf_estimation_method='Newton')

xgb_boost_model = xgb.XGBClassifier(learning_rate=0.098,
                                    n_estimators=350,
                                    max_depth = 5,
                                    subsample= 0.93,
                                    eval_metric='logloss')

model  = StackingClassifier(
    estimators=[('cat', cat_boost_model), ('xgb', xgb_boost_model)],
    final_estimator=LogisticRegression(),
    verbose=10,
    cv=3
)

class_weights=class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df[target_col]),
    y=df[target_col]
)

model.fit(x_train, y_train, sample_weight=class_weights[y_train])

x_train, y_train = RandomOverSampler().fit_resample(x_train, y_train)

prob = model.predict_proba(x_train)
prob = prob[:, 1]

y_pred, y_prob = (prob >= 0.5).astype(int), prob

print(model.final_estimator_.coef_)
roc_auc = roc_auc_score(y_train, y_prob)
f1 = f1_score(y_train, y_pred)

matrix = pd.DataFrame(confusion_matrix(y_train, y_pred), index=['TN', 'TP'], columns=['TN', 'TP'])

print(matrix)
print(f"Roc_auc Score: {roc_auc:.3f}")
print(f"F1 Score: {f1:.3f}")

with open("model.pickle", "wb") as file:
    pickle.dump({
        'model' : model
    }, file)
