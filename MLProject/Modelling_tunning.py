import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn

def plot_confusion(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(filename)
    plt.close()

def main():
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    mlflow.set_experiment("Titanic-Advanced-Tuning")
    mlflow.sklearn.autolog(log_models=True)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "processed_data")

    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

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
        random_state=42
    )

    with mlflow.start_run() as run:
        search.fit(X_train, y_train)

        y_pred = search.best_estimator_.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Manual logging 
        mlflow.log_metric("final_accuracy", acc)

        plot_confusion(y_test, y_pred, "confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        pd.DataFrame({
            "feature": X_train.columns,
            "importance": search.best_estimator_.feature_importances_
        }).to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        with open("run_info.txt", "w") as f:
            f.write(run.info.run_id)


if __name__ == "__main__":
    main()