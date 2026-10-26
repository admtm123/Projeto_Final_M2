import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


class ApplicationLifecycleTests(unittest.TestCase):
    def test_lifespan_loads_and_clears_model_artifacts(self):
        with (
            patch("main.initialize_model") as initialize_model,
            patch("main.clear_model_cache") as clear_model_cache,
            TestClient(app) as client,
        ):
            response = client.get("/openapi.json")
            self.assertEqual(response.status_code, 200)
            self.assertIn("/predict", response.json()["paths"])

        initialize_model.assert_called_once_with()
        clear_model_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
