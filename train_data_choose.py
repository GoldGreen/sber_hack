import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score,accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.utils import class_weight

df = pd.read_parquet('data/untrashed_data.parquet', engine='fastparquet')

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]


x_train, x_val, y_train, y_val = train_test_split(df[feature_cols], df[target_col], test_size=0.2)


xgb_model = xgb.XGBClassifier(learning_rate=0.035, max_depth = 5, subsample= 0.88, eval_metric='logloss')

param_grid = {
    'learning_rate':[0.025, 0.035, 0.045],
    'n_estimators':[500],
}

grid = GridSearchCV(xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=10, scoring='roc_auc')

class_weights=class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df[target_col]),
    y=df[target_col]
)

grid.fit(x_train, y_train, sample_weight=class_weights[y_train])

model = grid.best_estimator_
x_val, y_val = RandomUnderSampler().fit_resample(x_val, y_val)

y_pred = model.predict(x_val)

print(grid.best_params_)

print(f"[MODEL]: {type(model).__name__}")

roc_auc = roc_auc_score(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
matrix = pd.DataFrame(confusion_matrix(y_val, y_pred), index=['TN', 'TP'], columns=['TN', 'TP'])

print(matrix)
print(f"Roc_auc Score: {roc_auc:.3f}")
print(f"Accuracy Score: {accuracy:.3f}")
print(f"F1 Score: {f1:.3f}")