import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# MLflow Tracking Setup
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Titanic-Advanced-Tuning")

# autolog metrics and models
mlflow.sklearn.autolog(log_models=True)  

# Load Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

X_train = X_train.select_dtypes(include=["int64", "float64"]).astype(float)
X_test  = X_test.select_dtypes(include=["int64", "float64"]).astype(float)

# Artefak lokal manual
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Hyperparameter grid
n_estimators_list = [50, 100]
max_depth_list = [5, 10]

grid_results = []

# Run MLflow
with mlflow.start_run() as parent_run:
    mlflow.set_tag("project", "Titanic-Advanced-Tuning")
    mlflow.set_tag("author", "ekasandyaulia-lgtm")

    nested_runs_models = {}

    # Nested run untuk grid search
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            with mlflow.start_run(nested=True):
                # Log hyperparameters
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)

                # Train model
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                model.fit(X_train, y_train)

                # Metrics
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                mlflow.log_metric("accuracy", acc)

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                cm_path = os.path.join(ARTIFACT_DIR, f"cm_{n_estimators}_{max_depth}.png")
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d")
                plt.title(f"Confusion Matrix n={n_estimators} d={max_depth}")
                plt.savefig(cm_path)
                plt.close()
                mlflow.log_artifact(cm_path, artifact_path="manual")

                # Feature importance
                fi = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": model.feature_importances_
                })
                fi_path = os.path.join(ARTIFACT_DIR, f"fi_{n_estimators}_{max_depth}.csv")
                fi.to_csv(fi_path, index=False)
                mlflow.log_artifact(fi_path, artifact_path="manual")

                # Simpan ke grid_results
                grid_results.append({
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "accuracy": acc
                })

                # Simpan model sementara
                nested_runs_models[(n_estimators, max_depth)] = model

    # Simpan grid results
    grid_results_df = pd.DataFrame(grid_results)
    grid_results_path = os.path.join(ARTIFACT_DIR, "grid_results.csv")
    grid_results_df.to_csv(grid_results_path, index=False)
    mlflow.log_artifact(grid_results_path, artifact_path="manual")

    # Log model terbaik di parent run
    best_result = max(grid_results, key=lambda x: x["accuracy"])
    best_n_estimators = best_result["n_estimators"]
    best_max_depth = best_result["max_depth"]

    best_model = nested_runs_models[(best_n_estimators, best_max_depth)]
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="MLmodel",
        registered_model_name="Titanic_RF_Model"
    )

print("Model berhasil! RUN_ID induk:", parent_run.info.run_id)