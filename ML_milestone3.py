import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt


df = pd.read_csv('train_data.csv')

X = df.drop('Y', axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#Initialize the XGBoost
model = xgb.XGBClassifier(
    n_estimators=5000,
    learning_rate=0.005,
    max_depth=8,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric=['logloss', 'error'],
    early_stopping_rounds=500
)


eval_set = [(X_train, y_train), (X_test, y_test)]


print("Starting XGBoost training loop...")
model.fit(X_train, y_train, eval_set=eval_set, verbose=50)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n--- FINAL RESULTS ---")
print(f"Test Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))


results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 5))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Test/Validation')
plt.legend()
plt.ylabel('Log Loss')
plt.xlabel('Boosting Rounds (Loop iteration)')
plt.title('XGBoost Training Loop Progress')
plt.show()