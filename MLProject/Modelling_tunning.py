import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn

try:
    import dagshub
except ImportError:
    dagshub = None

def load_data(base_path="processed_data"):
    X_train = pd.read_csv(os.path.join(base_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(base_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(base_path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(base_path, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def plot_and_save_confusion(y_true, y_pred, out_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.matshow(cm)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main(args):
    # Set MLflow tracking URI
    tracking_uri = args.mlflow_tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Titanic-Advanced-Tuning")

    if args.use_dagshub and dagshub:
        token = os.environ.get("DAGSHUB_TOKEN")
        owner = os.environ.get("DAGSHUB_REPO_OWNER")
        repo = os.environ.get("DAGSHUB_REPO_NAME")
        if token and owner and repo:
            dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
        else:
            print("DagsHub requested but DAGSHUB_TOKEN or repo info missing. Skipping DagsHub init.")

    # Load dataset
    X_train, X_test, y_train, y_test = load_data(base_path="processed_data")

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    X_train = X_train[num_cols]
    X_test = X_test[num_cols]

    # Hyperparameter search
    clf = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    search = RandomizedSearchCV(
        clf, param_dist, n_iter=6, cv=3, random_state=42, n_jobs=1
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting MLflow run: {run_id}")

        # Fit model
        search.fit(X_train, y_train)
        best = search.best_estimator_
        best_params = search.best_params_

        # Predictions & accuracy
        y_pred = best.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Manual logging
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", float(acc))

        # Artefak tambahan 1: Confusion Matrix
        plot_and_save_confusion(y_test, y_pred, out_path="confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # Artefak tambahan 2: Feature importances
        try:
            fi = best.feature_importances_
            fi_df = pd.DataFrame({"feature": X_train.columns, "importance": fi})
            fi_df = fi_df.sort_values("importance", ascending=False)
            fi_df.to_csv("feature_importances.csv", index=False)
            mlflow.log_artifact("feature_importances.csv")
        except Exception as e:
            print("Feature importances not available:", e)

        # Log model
        mlflow.sklearn.log_model(best, artifact_path="model")

        # Save run_id to disk for CI workflows
        with open("run_info.txt", "w") as f:
            f.write(run_id)

        print("Training and logging complete. Run ID written to run_info.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--use-dagshub", default=False, action='store_true', help="Enable DagsHub logging")
    args = parser.parse_args()
    main(args)
