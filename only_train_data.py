import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

df = pd.read_parquet('data/untrashed_data.parquet', engine='fastparquet')

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]

# inputer =  KNNImputer(n_neighbors=1)
# df[feature_cols] = inputer.fit_transform(df[feature_cols].values)
# kmeans = KMeans(n_clusters=3, n_init=10)
# df['cluster'] = kmeans.fit_predict(df[feature_cols].values)
# df = pd.get_dummies(df, columns=['cluster'], prefix='cluster')
# feature_cols = [column for column in df.columns if column not in [target_col] ]

x_train, y_train = df[feature_cols], df[target_col]

model = CatBoostClassifier(verbose=False,
                           depth = 6,
                           l2_leaf_reg = 7,
                           iterations=700,
                           boosting_type='Plain',
                           bootstrap_type='Bernoulli',
                           subsample= 0.88,
                           rsm=0.88,
                           random_strength = 0.7,
                           leaf_estimation_iterations=5,
                           max_ctr_complexity = 1,
                           leaf_estimation_method='Newton',
                           auto_class_weights = 'Balanced')


model.fit(x_train, y_train)

with open("model.pickle", "wb") as file:
    pickle.dump({
        'cluster_model': None,
        'inputer': None,
        'model' : model
    }, file)
