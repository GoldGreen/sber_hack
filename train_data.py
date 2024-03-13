import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgbm
from sklearn.utils import class_weight

df = pd.read_parquet('data/untrashed_data.parquet', engine='fastparquet')

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]


x_train, x_val, y_train, y_val = train_test_split(df[feature_cols], df[target_col], test_size=0.4)

cat_boost_model = CatBoostClassifier(verbose=False,
                           depth = 6,
                           l2_leaf_reg = 7,
                           learning_rate=0.08,
                           iterations=250,
                           boosting_type='Plain',
                           bootstrap_type='Bernoulli',
                           subsample= 0.93,
                           rsm=0.88,
                           random_strength = 0.8,
                           leaf_estimation_iterations=5,
                           max_ctr_complexity = 1,
                           leaf_estimation_method='Newton')

xgb_boost_model = xgb.XGBClassifier(learning_rate=0.05, 
                                    n_estimators=200, 
                                    eval_metric='logloss')



lgbm_model = lgbm.LGBMClassifier(objective= 'binary',
                        max_depth= 5,
                        num_leaves=20,
                        learning_rate= 0.04, 
                        n_estimators=150)

model  = StackingClassifier(
    estimators=[('cat', cat_boost_model), ('xgb', xgb_boost_model), ('lgbm', lgbm_model)],
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

x_val, y_val = RandomUnderSampler().fit_resample(x_val, y_val)

y_pred = model.predict(x_val)
y_pred_prob = model.predict_proba(x_val)[:,1]

print(f"[MODEL]: {type(model).__name__}")

roc_auc = roc_auc_score(y_val, y_pred_prob)
f1 = f1_score(y_val, y_pred)
matrix = pd.DataFrame(confusion_matrix(y_val, y_pred), index=['TN', 'TP'], columns=['TN', 'TP'])

print(matrix)
print(f"Roc_auc Score: {roc_auc:.3f}")
print(f"F1 Score: {f1:.3f}")