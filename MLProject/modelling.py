import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from mlflow.models.signature import infer_signature

# MLflow Tracking Setup
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Titanic-Advanced-Tuning")

# Dataset Loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

X_train = X_train.select_dtypes(include=["int64", "float64"])
X_test  = X_test.select_dtypes(include=["int64", "float64"])

# Folder artefak
os.makedirs("artifacts", exist_ok=True)

# Hyperparameter Grid
n_estimators_list = [50, 100]
max_depth_list = [5, 10]

# Training Loop
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:

        with mlflow.start_run():

            # Log parameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", acc)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title(f"Confusion Matrix n={n_estimators} d={max_depth}")
            cm_path = f"artifacts/cm_{n_estimators}_{max_depth}.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)

            # Feature Importance
            fi = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_
            })
            fi_path = f"artifacts/fi_{n_estimators}_{max_depth}.csv"
            fi.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path)

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.iloc[:5]
            )

print("model berhasil!")