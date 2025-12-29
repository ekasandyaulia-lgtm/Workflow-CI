import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn

try:
    import dagshub
except ImportError:
    dagshub = None

def plot_confusion(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

def main(args):
    # Setup MLflow tracking
    tracking_uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Titanic-Advanced-Tuning")

    if args.use_dagshub and dagshub:
        token = os.environ.get("DAGSHUB_TOKEN")
        owner = os.environ.get("DAGSHUB_REPO_OWNER")
        repo = os.environ.get("DAGSHUB_REPO_NAME")
        if token and owner and repo:
            dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)

    # Load dataset
    X_train = pd.read_csv("processed_data/X_train.csv")
    X_test = pd.read_csv("processed_data/X_test.csv")
    y_train = pd.read_csv("processed_data/y_train.csv").values.ravel()
    y_test = pd.read_csv("processed_data/y_test.csv").values.ravel()

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    X_train = X_train[num_cols]
    X_test = X_test[num_cols]

    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Train model
        params = {"C": 1.0, "max_iter": 200, "solver": "liblinear"}
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Predict & evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log params & metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        # Log artifacts
        plot_confusion(y_test, y_pred, "confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        coef_df = pd.DataFrame(model.coef_, columns=X_train.columns)
        coef_df.to_csv("model_coefficients.csv", index=False)
        mlflow.log_artifact("model_coefficients.csv")

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="TitanicModel")

        # Save run_id for CI
        with open("run_info.txt", "w") as f:
            f.write(run_id)

    print("Training and logging complete. Run ID saved in run_info.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--use-dagshub", action="store_true")
    args = parser.parse_args()
    main(args)
