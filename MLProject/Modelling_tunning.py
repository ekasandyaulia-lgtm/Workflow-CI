import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn


def plot_confusion(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()


def main():
    # MLflow tracking
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    mlflow.set_experiment("Titanic-Advanced-Tuning")

    # Autolog (wajib untuk artefak model/)
    mlflow.sklearn.autolog(log_models=True)

    # Load processed data
    X_train = pd.read_csv("processed_data/X_train.csv")
    X_test = pd.read_csv("processed_data/X_test.csv")
    y_train = pd.read_csv("processed_data/y_train.csv").values.ravel()
    y_test = pd.read_csv("processed_data/y_test.csv").values.ravel()

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    X_train = X_train[num_cols]
    X_test = X_test[num_cols]

    model = RandomForestClassifier(random_state=42)

    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        random_state=42,
        n_jobs=1
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("final_accuracy", acc)

        plot_confusion(y_test, y_pred)
        mlflow.log_artifact("confusion_matrix.png")

        fi = pd.DataFrame({
            "feature": X_train.columns,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)

        fi.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        with open("run_info.txt", "w") as f:
            f.write(run_id)


if __name__ == "__main__":
    main()