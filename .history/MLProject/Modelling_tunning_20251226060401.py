import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import os

mlflow.set_tracking_uri(
    "https://dagshub.com/ekasandyaulia-lgtm/SMLS_Eka_Sandy_Aulia_Puspitasari.mlflow"
)

dagshub.init(
    repo_owner="ekasandyaulia-lgtm",
    repo_name="SMLS_Eka_Sandy_Aulia_Puspitasari",
    mlflow=True
)

mlflow.set_experiment("Titanic-Advanced-Tuning")

# no auto log
mlflow.autolog(disable=True)

# Load data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

X_train = pd.read_csv(os.path.join(BASE_DIR, "processed_data", "X_train.csv"))
X_test  = pd.read_csv(os.path.join(BASE_DIR, "processed_data", "X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE_DIR, "processed_data", "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(BASE_DIR, "processed_data", "y_test.csv")).values.ravel()

X_train = X_train.select_dtypes(include=["int64", "float64"])
X_test  = X_test.select_dtypes(include=["int64", "float64"])

# Hyperparameter loop
n_estimators_list = [50, 100]
max_depth_list = [5, 10]

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:

        with mlflow.start_run():

            # log parameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("model", "RandomForestClassifier")

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # log metric
            mlflow.log_metric("accuracy", acc)

            # Artifact 1: Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure()
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title("Confusion Matrix")

            os.makedirs("artifacts", exist_ok=True)
            cm_path = f"artifacts/confusion_matrix_{n_estimators}_{max_depth}.png"
            plt.savefig(cm_path)
            plt.close()

            mlflow.log_artifact(cm_path)

            # Artifact 2: Feature Importance
            fi = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_
            }).sort_values(by="importance", ascending=False)

            fi_path = f"artifacts/feature_importance_{n_estimators}_{max_depth}.csv"
            fi.to_csv(fi_path, index=False)

            mlflow.log_artifact(fi_path)

            print(f"Run selesai | accuracy={acc}")