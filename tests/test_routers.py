import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import router
from api.schemas import PredictionRequest, PredictionResponse


class PredictionRouterTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_predict_returns_the_service_response(self):
        payload = {
            name: index
            for index, name in enumerate(PredictionRequest.model_fields, start=1)
        }
        expected = PredictionResponse(
            preco_estimado=123_456.0,
            versao_modelo="v1",
        )

        with patch("api.routers.predict_price", return_value=expected) as predict:
            response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected.model_dump())
        predict.assert_called_once()

    def test_predict_rejects_incomplete_request(self):
        response = self.client.post("/predict", json={})

        self.assertEqual(response.status_code, 422)

    def test_predict_rejects_extra_fields(self):
        payload = {
            name: 1
            for name in PredictionRequest.model_fields
        }
        payload["unexpected"] = 1

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
