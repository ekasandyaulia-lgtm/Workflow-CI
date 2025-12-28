import os
import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

mlflow.autolog(
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True
)

assert os.getenv("DAGSHUB_USER_TOKEN") is not None, "DAGSHUB_USER_TOKEN missing"

dagshub.init(
    repo_owner="ekasandyaulia-lgtm",
    repo_name="SMLS_Eka_Sandy_Aulia_Puspitasari",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/ekasandyaulia-lgtm/"
    "SMLS_Eka_Sandy_Aulia_Puspitasari.mlflow"
)

mlflow.set_experiment("Titanic-Advanced-Tuning")

X_train = pd.read_csv("processed_data/X_train.csv")
X_test  = pd.read_csv("processed_data/X_test.csv")
y_train = pd.read_csv("processed_data/y_train.csv").values.ravel()
y_test  = pd.read_csv("processed_data/y_test.csv").values.ravel()

X_train = X_train.select_dtypes(include=["int64", "float64"])
X_test  = X_test.select_dtypes(include=["int64", "float64"])

with mlflow.start_run():

    n_estimators = 100
    random_state = 42

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(X_train, y_train)

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

print("MLflow Advanced Training Completed")
