import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data hasil preprocessing
X_train = pd.read_csv("processed_data/X_train.csv")
X_test = pd.read_csv("processed_data/X_test.csv")
y_train = pd.read_csv("processed_data/y_train.csv").values.ravel()
y_test = pd.read_csv("processed_data/y_test.csv").values.ravel()

# Data numeric
X_train = X_train.select_dtypes(include=["int64", "float64"])
X_test = X_test.select_dtypes(include=["int64", "float64"])

mlflow.set_experiment("Titanic-Basic-Model")

with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)