import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
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

with open("columns.txt") as file:
    feature_cols = [line.rstrip() for line in file]
target_col = 'target'

x_train, y_train = df[feature_cols], df[target_col]

cat_boost_ratio = 1
cat_boost_model = CatBoostClassifier(verbose=False,
                           max_depth=6,
                           l2_leaf_reg = 7,
                           learning_rate = 0.046 * cat_boost_ratio,
                           iterations=int(700 / cat_boost_ratio),
                           boosting_type='Plain',
                           bootstrap_type='Bernoulli',
                           subsample= 0.88,
                           rsm=0.88,
                           random_strength=0.7,
                           leaf_estimation_method='Newton')

xgb_ratio = 1
xgb_boost_model = xgb.XGBClassifier(learning_rate=0.048 * xgb_ratio,
                                    n_estimators=int(700 / xgb_ratio),
                                    max_depth=5,
                                    reg_alpha=7,
                                    reg_lambda=1, 
                                    subsample=0.88,
                                    colsample_bytree=0.88,
                                    eval_metric='logloss')

model  = StackingClassifier(
    estimators=[('cat', cat_boost_model), ('xgb', xgb_boost_model)],
    final_estimator=LogisticRegression(),
    verbose=10,
    cv=5
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
