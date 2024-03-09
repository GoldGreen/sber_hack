import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

df = pd.read_parquet('data/processed_data.parquet', engine='fastparquet')

target_col = 'target'
feature_cols = [column for column in df.columns if column not in [target_col] ]

abs_true = (df[target_col] == 1).sum() /  df.shape[0]
abs_false = 1 - abs_true

print(f'{abs_true} {abs_false}')

y = df[target_col] 
x = df[feature_cols]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression(class_weight={0: 1 - abs_false, 1: 1 - abs_true}, max_iter=200)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")