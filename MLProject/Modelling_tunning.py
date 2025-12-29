import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn

DAGSHUB_TOKEN = os.environ["DAGSHUB_TOKEN"]

# Set tracking URI dan experiment
mlflow.set_tracking_uri(
    f"https://{DAGSHUB_TOKEN}@dagshub.com/ekasandyaulia-lgtm/SMLS_Eka_Sandy_Aulia_Puspitasari.mlflow"
)
mlflow.set_experiment("Titanic-Advanced-Tuning")

# Load dataset
X_train = pd.read_csv("processed_data/X_train.csv")
X_test  = pd.read_csv("processed_data/X_test.csv")
y_train = pd.read_csv("processed_data/y_train.csv").values.ravel()
y_test  = pd.read_csv("processed_data/y_test.csv").values.ravel()

# Pilih kolom numerik
X_train = X_train.select_dtypes(include=["int64", "float64"])
X_test  = X_test.select_dtypes(include=["int64", "float64"])

with mlflow.start_run() as run:
    RUN_ID = run.info.run_id
    print(f"MLflow Run ID: {RUN_ID}")

    # Model
    params = {"C": 1.0, "max_iter": 200, "solver": "liblinear"}
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log parameter dan metric
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)

    # Log artefak: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Confusion Matrix (Accuracy: {acc:.4f})")
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # Log artefak: Koefisien model
    coef_df = pd.DataFrame(model.coef_, columns=X_train.columns)
    coef_df.to_csv("model_coefficients.csv", index=False)
    mlflow.log_artifact("model_coefficients.csv")

    # Log model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="TitanicModel"
    )

print("Training selesai dan artefak tersimpan di MLflow / DagsHub.")
