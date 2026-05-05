import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data():
    data = load_iris()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc


def save_model(model, path="model.pkl"):
    joblib.dump(model, path)


def main():
    mlflow.set_experiment("iris-classifier")

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_data()

        model = train_model(X_train, y_train)
        acc = evaluate(model, X_test, y_test)

        save_model(model)

        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model Accuracy: {acc}")


if __name__ == "__main__":
    main()