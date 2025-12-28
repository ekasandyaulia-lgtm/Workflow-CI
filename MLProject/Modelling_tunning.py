import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

dagshub.init(
    repo_owner="ekasandyaulia-lgtm",
    repo_name="SMLS_Eka_Sandy_Aulia_Puspitasari",
    mlflow=True
)

# Tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/ekasandyaulia-lgtm/SMLS_Eka_Sandy_Aulia_Puspitasari.mlflow"
)

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_TOKEN")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

mlflow.set_experiment("Titanic-Advanced-Tuning")

# Autolog
mlflow.sklearn.autolog(log_models=True)

# Data
X_train = pd.read_csv("processed_data/X_train.csv")
X_test  = pd.read_csv("processed_data/X_test.csv")
y_train = pd.read_csv("processed_data/y_train.csv").values.ravel()
y_test  = pd.read_csv("processed_data/y_test.csv").values.ravel()

X_train = X_train.select_dtypes(include=["int64", "float64"])
X_test  = X_test.select_dtypes(include=["int64", "float64"])

# Train 
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Metrics
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
mlflow.log_metric("accuracy", acc)

# Artifact: confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig("confusion_matrix.png")
plt.close()
mlflow.log_artifact("confusion_matrix.png")

# Artifact: feature importance
fi = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

fi.to_csv("feature_importance.csv", index=False)
mlflow.log_artifact("feature_importance.csv")

print("MLflow Project training completed successfully")
