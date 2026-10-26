import unittest
from unittest.mock import patch

import numpy as np

from api.schemas import PredictionRequest
from api.services import _load_artifacts, predict_price


class _FakeScaler:
    n_features_in_ = len(PredictionRequest.model_fields)
    feature_names_in_ = np.array(list(PredictionRequest.model_fields))

    def transform(self, values):
        self.values = values
        return values.to_numpy()


class _FakeModel:
    n_features_in_ = len(PredictionRequest.model_fields)

    def predict(self, values):
        self.values = values
        return np.array([123_456.0])


class PredictPriceTests(unittest.TestCase):
    def setUp(self):
        _load_artifacts.cache_clear()
        self.scaler = _FakeScaler()
        self.model = _FakeModel()
        self.artifacts = iter((self.model, self.scaler))
        self.loader = patch(
            "api.services.joblib.load",
            side_effect=lambda _path: next(self.artifacts),
        )
        self.loader.start()

    def tearDown(self):
        self.loader.stop()
        _load_artifacts.cache_clear()

    def test_prediction_uses_metadata_feature_order_and_scaler(self):
        payload = {
            name: index
            for index, name in enumerate(PredictionRequest.model_fields, start=1)
        }

        result = predict_price(PredictionRequest(**payload))

        self.assertEqual(result.preco_estimado, 123_456.0)
        self.assertEqual(result.versao_modelo, "v1")
        self.assertEqual(
            list(self.scaler.values.columns),
            list(PredictionRequest.model_fields),
        )
        self.assertEqual(self.model.values.shape, (1, 23))


if __name__ == "__main__":
    unittest.main()
