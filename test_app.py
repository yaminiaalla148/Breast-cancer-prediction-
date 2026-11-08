import unittest

import pandas as pd

from app import create_app


class ExampleModel:
    classes_ = [0, 1]

    def __init__(self):
        self.prediction = 1

    def predict(self, features):
        return [self.prediction]

class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.feature_names = ["mean radius", "radius error", "worst radius"]
        self.model = ExampleModel()
        self.application = create_app(
            {
                "feature_names": self.feature_names,
                "model": self.model,
                "results": pd.DataFrame(
                    [{"Model": "SVM", "Accuracy": 0.98}]
                ),
                "accuracy": 0.98,
                "examples": {
                    "benign": {
                        "mean radius": 10.0,
                        "radius error": 0.1,
                        "worst radius": 12.0,
                    },
                    "malignant": {
                        "mean radius": 18.0,
                        "radius error": 1.0,
                        "worst radius": 23.0,
                    },
                },
            }
        )
        self.client = self.application.test_client()

    def test_dashboard_renders_input_form(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Enter measurements", response.data)
        self.assertIn(b"mean radius", response.data)
        self.assertIn(b"Radius Error", response.data)

    def test_prediction_returns_class(self):
        response = self.client.post(
            "/predict",
            data={
                "mean radius": "10",
                "radius error": "0.1",
                "worst radius": "12",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No cancer detected", response.data)
        self.assertIn(b"cannot diagnose or rule out cancer", response.data)
        self.assertIn(b'id="prediction-result"', response.data)
        self.assertIn(b'role="status"', response.data)

    def test_malignant_prediction_is_explained_clearly(self):
        self.model.prediction = 0
        response = self.client.post(
            "/predict",
            data={
                "mean radius": "18",
                "radius error": "1",
                "worst radius": "23",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Cancer detected", response.data)
        self.assertIn(b"malignant (cancer-associated) pattern", response.data)

    def test_prediction_rejects_invalid_values(self):
        response = self.client.post(
            "/predict",
            data={
                "mean radius": "-1",
                "radius error": "0.1",
                "worst radius": "not a number",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Enter a valid, non-negative number", response.data)

    def test_examples_are_available(self):
        response = self.client.get("/api/examples")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["benign"]["mean radius"], 10.0)
        self.assertEqual(response.json["benign"]["radius error"], 0.1)


if __name__ == "__main__":
    unittest.main()
