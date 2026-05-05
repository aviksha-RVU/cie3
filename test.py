"""
Unit tests for main.py model functions.
"""

import unittest
from main import load_data, train_model, evaluate


class TestModel(unittest.TestCase):
    """Test cases for ML model."""

    def test_data_loading(self):
        """Test if data loads correctly."""
        x_train, x_test, y_train, y_test = load_data()

        self.assertTrue(len(x_train) > 0)
        self.assertEqual(len(x_train), len(y_train))
        self.assertTrue(len(x_test) > 0)
        self.assertTrue(len(y_test) > 0)

    def test_training(self):
        """Test model training."""
        x_train, _, y_train, _ = load_data()
        model = train_model(x_train, y_train)

        self.assertIsNotNone(model)

    def test_evaluation(self):
        """Test model evaluation."""
        x_train, x_test, y_train, y_test = load_data()
        model = train_model(x_train, y_train)
        acc = evaluate(model, x_test, y_test)

        self.assertTrue(0 <= acc <= 1)


if __name__ == "__main__":
    unittest.main()
