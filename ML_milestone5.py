import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)
# Load Dataset
train = pd.read_csv('train_data.csv') 
test = pd.read_csv('test_data.csv')

# Split Features And Labels
X_test = test.drop('Y', axis=1)
y_test = test['Y']

# Create Optimized Model
model = xgb.XGBClassifier(
    learning_rate=0.01,
    n_estimators=500,
    max_depth=4,
    reg_alpha=0.5,
    reg_lambda=2.0,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    eval_metric='logloss',
    random_state=42
)

 
# Train Model
X_train = train.drop('Y', axis=1) 
y_train = train['Y'] 
X_test = test.drop('Y', axis=1) 
y_test = test['Y']
model.fit(X_train, y_train)


# Predict Test Set
y_pred = model.predict(X_test)

# Probability predictions for ROC & PR curves
y_prob = model.predict_proba(X_test)[:, 1]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", round(accuracy * 100, 2), "%")

# ADDITIONAL METRICS
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall_curve, precision_curve)

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))

# CLASSIFICATION REPORT
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n")
print(cm)

# PLOT CONFUSION MATRIX
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Test Set")
plt.show()

# ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.4f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()


# PRECISION-RECALL CURVE
plt.figure(figsize=(6,6))
plt.plot(recall_curve, precision_curve,
         label=f'PR-AUC = {pr_auc:.4f}')

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.legend()

plt.show()


# BAR CHART OF METRICS
metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'ROC-AUC': roc_auc,
    'PR-AUC': pr_auc
}

names = list(metrics.keys())
values = list(metrics.values())

plt.figure(figsize=(8,5))

bars = plt.bar(names, values)

plt.ylim(0,1)

plt.ylabel("Score")

plt.title("Model Performance Metrics")

# Add values on top
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 0.01,
        f'{yval:.2f}',
        ha='center'
    )

plt.show()