import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.routers import router
from api.schemas import PredictionRequest, PredictionResponse


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


class PredictionRouterTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_predict_returns_the_service_response(self):
        payload = _valid_payload()
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
        payload = _valid_payload()
        payload["unexpected"] = 1

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 422)

    def test_predict_rejects_text_in_numeric_field(self):
        payload = _valid_payload()
        payload["sqft_living"] = "1500"

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 422)

    def test_predict_rejects_out_of_range_fields(self):
        for field, value in (("view", 5), ("waterfront", 2), ("lat", 10)):
            with self.subTest(field=field):
                payload = _valid_payload()
                payload[field] = value
                response = self.client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 422)

    def test_response_schema_rejects_invalid_output(self):
        for invalid_response in (
            {"preco_estimado": 0.0, "versao_modelo": "v1"},
            {"preco_estimado": 1.0, "versao_modelo": ""},
            {"preco_estimado": "123.0", "versao_modelo": "v1"},
        ):
            with self.subTest(response=invalid_response):
                with self.assertRaises(ValidationError):
                    PredictionResponse.model_validate(invalid_response)


if __name__ == "__main__":
    unittest.main()
