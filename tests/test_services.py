import unittest
from unittest.mock import patch

import numpy as np

from api.schemas import PredictionRequest
from api.services import (
    clear_model_cache,
    initialize_model,
    predict_price,
)


def _valid_payload():
    payload = {name: 0 for name in PredictionRequest.model_fields}
    payload.update(
        bedrooms=3,
        bathrooms=2.5,
        sqft_living=1_500,
        sqft_lot=5_000,
        floors=1.5,
        condition=3,
        grade=7,
        idade_imovel=50,
        lat=47.5,
        long=-122.2,
        zipcode_faixa_faixa_1=1,
    )
    return payload


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
        clear_model_cache()
        self.scaler = _FakeScaler()
        self.model = _FakeModel()
        self.artifacts = iter((self.model, self.scaler))
        self.loader = patch(
            "api.services.joblib.load",
            side_effect=lambda _path: next(self.artifacts),
        )
        self.joblib_load = self.loader.start()
        initialize_model()

    def tearDown(self):
        self.loader.stop()
        clear_model_cache()

    def test_prediction_uses_metadata_feature_order_and_scaler(self):
        payload = _valid_payload()

        result = predict_price(PredictionRequest(**payload))

        self.assertEqual(result.preco_estimado, 123_456.0)
        self.assertEqual(result.versao_modelo, "v1")
        self.assertEqual(
            list(self.scaler.values.columns),
            list(PredictionRequest.model_fields),
        )
        self.assertEqual(self.model.values.shape, (1, 23))
        predict_price(PredictionRequest(**payload))
        self.assertEqual(self.joblib_load.call_count, 2)
        initialize_model()
        self.assertEqual(self.joblib_load.call_count, 2)

    def test_prediction_before_initialization_does_not_read_artifacts(self):
        clear_model_cache()
        self.joblib_load.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "lifespan do FastAPI"):
            predict_price(PredictionRequest(**_valid_payload()))
        self.joblib_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
