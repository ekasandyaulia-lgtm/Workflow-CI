import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# dagshub integration
os.environ["MLFLOW_TRACKING_USERNAME"] = "ekasandyaulia-lgtm"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(
    "https://dagshub.com/ekasandyaulia-lgtm/SMLS_Eka_Sandy_Aulia_Puspitasari.mlflow"
)

mlflow.set_experiment("Titanic-Advanced-Tuning")

# load data hasil preprocessing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

X_train = pd.read_csv(os.path.join(BASE_DIR, "processed_data", "X_train.csv"))
X_test  = pd.read_csv(os.path.join(BASE_DIR, "processed_data", "X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE_DIR, "processed_data", "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(BASE_DIR, "processed_data", "y_test.csv")).values.ravel()

X_train = X_train.select_dtypes(include=["int64", "float64"])
X_test  = X_test.select_dtypes(include=["int64", "float64"])

# hyperparameter tuning
n_estimators_list = [50, 100]
max_depth_list = [5, 10]

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:

        with mlflow.start_run():

            # manual logging params
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # manual logging metric
            mlflow.log_metric("accuracy", acc)

            # artifact 1: Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title("Confusion Matrix")

            os.makedirs("artifacts", exist_ok=True)
            cm_path = f"artifacts/cm_{n_estimators}_{max_depth}.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)

            # artifact 2: Feature Importance
            fi = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_
            }).sort_values(by="importance", ascending=False)

            fi_path = f"artifacts/fi_{n_estimators}_{max_depth}.csv"
            fi.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path)

            #log model
            mlflow.sklearn.log_model(model, "model")

            print(f"Run selesai | acc={acc}")