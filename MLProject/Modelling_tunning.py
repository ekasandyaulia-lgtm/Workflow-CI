import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True
)

# Load data
X_train = pd.read_csv("processed_data/X_train.csv")
X_test  = pd.read_csv("processed_data/X_test.csv")
y_train = pd.read_csv("processed_data/y_train.csv").values.ravel()
y_test  = pd.read_csv("processed_data/y_test.csv").values.ravel()

X_train = X_train.select_dtypes(include=["int64", "float64"])
X_test  = X_test.select_dtypes(include=["int64", "float64"])

assert not X_train.empty, "X_train kosong!"
assert not X_test.empty, "X_test kosong!"

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

mlflow.log_metric("test_accuracy", acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title(f"Confusion Matrix (Accuracy: {acc:.4f})")
plt.savefig("confusion_matrix.png")
plt.close()

mlflow.log_artifact("confusion_matrix.png")

# Feature Importance
fi = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

fi.to_csv("feature_importance.csv", index=False)
mlflow.log_artifact("feature_importance.csv")

mlflow.set_tag("stage", "advanced")
mlflow.set_tag("model", "RandomForest")

print("MLflow Advanced Training Completed")
