import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tqdm.auto import tqdm

df = pd.read_parquet('data/processed_data.parquet', engine='fastparquet')

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]

y = df[target_col] 
x = df[feature_cols]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

models = [
    GradientBoostingClassifier(),
    RandomForestClassifier()
]

max_features = ['sqrt',  25]

gb_param_grid = { 'n_estimators': [50, 100, 200], 'learning_rate':[0.01], 'max_features':max_features}
rf_param_grid = { 'n_estimators': [100, 200, 300], 'max_features':max_features}

param_grids = [gb_param_grid,rf_param_grid]

for model, param_grid in zip(models,param_grids):
    grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, verbose=10)
    grid_search.fit(x_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    y_pred = best_model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print()
    print(f'Model: {type(model).__name__}, best params: {best_params}')
    print(f"Accuracy Score: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Roc_auc Score: {roc_auc:.3f}")
    print()