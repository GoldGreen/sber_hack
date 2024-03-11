import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

df = pd.read_parquet('data/processed_data.parquet', engine='fastparquet')

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]

kmeans = KMeans(n_clusters=9, n_init='auto')
df['cluster'] = kmeans.fit_predict(df[feature_cols].values)
df = pd.get_dummies(df, columns=['cluster'], prefix='cluster')
feature_cols = [column for column in df.columns if column not in [target_col] ]

x_train, x_val, y_train, y_val = train_test_split(df[feature_cols], df[target_col], test_size=0.05, random_state=30)

model = CatBoostClassifier(verbose=False,
                           auto_class_weights = 'Balanced',  
                           depth = 6,    
                           l2_leaf_reg = 7,    
                           learning_rate = 0.1,    
                           n_estimators = 300,    
                           bagging_temperature = 0.5,    
                           border_count = 256,   
                           random_strength = 0.7,
                           subsample= 0.8,
                           max_ctr_complexity = 1)

model.fit(x_train, y_train)

x_val, y_val = RandomOverSampler().fit_resample(x_val, y_val)

y_pred = model.predict(x_val)

roc_auc = roc_auc_score(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
matrix = pd.DataFrame(confusion_matrix(y_val, y_pred), index=['TN', 'TP'], columns=['TN', 'TP'])

print(matrix)
print(f"Roc_auc Score: {roc_auc:.3f}")
print(f"Accuracy Score: {accuracy:.3f}")
print(f"F1 Score: {f1:.3f}")

with open("model.pickle", "wb") as file:
    pickle.dump({
        'cluster_model': kmeans,
        'model' : model
    }, file)
