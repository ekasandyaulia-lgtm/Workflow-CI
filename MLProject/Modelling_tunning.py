import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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
    # MLflow tracking
    tracking_uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Titanic-Advanced-Tuning")

    # Optional DagsHub init
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

    # Keep numeric columns only
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    X_train = X_train[num_cols]
    X_test = X_test[num_cols]

    # Hyperparameter tuning + MLflow run
    clf = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    search = RandomizedSearchCV(clf, param_dist, n_iter=6, cv=3, random_state=42, n_jobs=1)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Fit model
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_

        # Predict & evaluate
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log params & metric
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", float(acc))

        # Log artifacts
        plot_confusion(y_test, y_pred)
        mlflow.log_artifact("confusion_matrix.png")

        fi_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)
        fi_df.to_csv("feature_importances.csv", index=False)
        mlflow.log_artifact("feature_importances.csv")

        # Log model
        mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name="TitanicModel")

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
