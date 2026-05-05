import unittest
from main import load_data, train_model, evaluate


class TestModel(unittest.TestCase):

    def test_data_loading(self):
        X_train, X_test, y_train, y_test = load_data()
        self.assertTrue(len(X_train) > 0)
        self.assertEqual(len(X_train), len(y_train))

    def test_training(self):
        X_train, X_test, y_train, y_test = load_data()
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)

    def test_evaluation(self):
        X_train, X_test, y_train, y_test = load_data()
        model = train_model(X_train, y_train)
        acc = evaluate(model, X_test, y_test)
        self.assertTrue(acc >= 0 and acc <= 1)


if __name__ == "__main__":
    unittest.main()