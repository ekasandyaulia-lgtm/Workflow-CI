import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

dagshub.init(
    repo_owner="ekasandyaulia-lgtm",
    repo_name="SMLS_Eka_Sandy_Aulia_Puspitasari",
    mlflow=True
)

mlflow.set_experiment("Titanic-Advanced-Tuning")

# load data
X_train = pd.read_csv("processed_data/X_train.csv")
X_test = pd.read_csv("processed_data/X_test.csv")
y_train = pd.read_csv("processed_data/y_train.csv").values.ravel()
y_test = pd.read_csv("processed_data/y_test.csv").values.ravel()

with mlflow.start_run(run_name="LogReg-Advanced"):

    params = {
        "C": 1.0,
        "max_iter": 200,
        "solver": "liblinear"
    }

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # log parameter dan metrik
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)

    # artefak tambahan 1: confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # artefak tambahan 2: koefisien model
    coef_df = pd.DataFrame(model.coef_, columns=X_train.columns)
    coef_df.to_csv("model_coefficients.csv", index=False)
    mlflow.log_artifact("model_coefficients.csv")

    # log model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="TitanicModel"
    )

print("training selesai dan artefak tersimpan")
