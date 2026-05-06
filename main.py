"""
Main ML pipeline using Iris dataset with MLflow logging.
"""
print("MODEL VERSION: v2")
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data():
    """Load and split Iris dataset."""
    x, y = load_iris(return_X_y=True)
    return train_test_split(x, y, test_size=0.2, random_state=42)


def train_model(x_train, y_train):
    """Train RandomForest model."""
    model = RandomForestClassifier(n_estimators=100)
    model.fit(x_train, y_train)
    return model


def evaluate(model, x_test, y_test):
    """Evaluate model accuracy."""
    predictions = model.predict(x_test)
    return accuracy_score(y_test, predictions)


def save_model(model):
    """Save trained model."""
    joblib.dump(model, "model.pkl")


def main():
    """Main execution pipeline."""
    mlflow.set_experiment("iris-classifier")

    with mlflow.start_run():
        x_train, x_test, y_train, y_test = load_data()

        model = train_model(x_train, y_train)
        accuracy = evaluate(model, x_test, y_test)

        save_model(model)

        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
