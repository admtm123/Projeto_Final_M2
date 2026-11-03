"""Testes HTTP das rotas de monitoramento e previsão."""

import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.exception_handlers import register_exception_handlers
from api.routers import router
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse


def _valid_payload():
    """Monta um exemplo válido com todos os campos exigidos pelo schema."""
    # Inicializa todas as features com zero; depois define valores realistas
    # para os campos cujas regras não permitem zero.
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
    """Verifica contratos HTTP, validação de entrada e tratamento de erros."""

    def setUp(self):
        # Cria uma aplicação pequena com as mesmas rotas e handlers de erro,
        # isolando estes testes da inicialização real do modelo.
        self.app = FastAPI()
        register_exception_handlers(self.app)
        self.app.include_router(router)
        self.client = TestClient(self.app)

    def test_health_returns_200_with_loaded_model_state(self):
        # Simula os artefatos disponíveis para verificar o formato de sucesso.
        health = HealthResponse(
            status="ok",
            service_active=True,
            model_loaded=True,
            scaler_loaded=True,
        )

        with patch("api.routers.get_health_status", return_value=health):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), health.model_dump())

    def test_health_returns_200_when_model_is_not_loaded(self):
        # Mesmo degradada, a checagem responde 200: o corpo informa que o
        # serviço está ativo, mas que os artefatos ainda não estão carregados.
        health = HealthResponse(
            status="degraded",
            service_active=True,
            model_loaded=False,
            scaler_loaded=False,
        )

        with patch("api.routers.get_health_status", return_value=health):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["service_active"])
        self.assertFalse(response.json()["model_loaded"])

    def test_predict_returns_the_service_response(self):
        # Substitui o serviço para verificar que a rota encaminha a entrada
        # e devolve o contrato de saída sem executar inferência real.
        payload = _valid_payload()
        expected = PredictionResponse(
            preco_estimado=123_456.79,
            versao_modelo="v1",
        )

        with patch("api.routers.predict_price", return_value=expected) as predict:
            response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected.model_dump())
        predict.assert_called_once()

    def test_predict_rejects_incomplete_request(self):
        # O schema exige as features necessárias; um corpo incompleto deve
        # ser recusado pelo FastAPI/Pydantic antes de chamar o serviço.
        response = self.client.post("/predict", json={})

        self.assertEqual(response.status_code, 422)

    def test_predict_rejects_extra_fields(self):
        # Campos fora do contrato são proibidos para detectar erros no payload.
        payload = _valid_payload()
        payload["unexpected"] = 1

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 422)

    def test_predict_rejects_text_in_numeric_field(self):
        # A validação estrita não converte texto numérico em número.
        payload = _valid_payload()
        payload["sqft_living"] = "1500"

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 422)

    def test_predict_rejects_out_of_range_fields(self):
        # Confere diferentes limites de domínio definidos no schema.
        for field, value in (("view", 5), ("waterfront", 2), ("lat", 10)):
            with self.subTest(field=field):
                payload = _valid_payload()
                payload[field] = value
                response = self.client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 422)

    def test_predict_returns_400_for_malformed_json(self):
        # JSON sintaticamente inválido é diferente de um JSON válido que viola
        # o schema, por isso o handler deve responder 400 em vez de 422.
        response = self.client.post(
            "/predict",
            content='{"bedrooms":',
            headers={"content-type": "application/json"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("malformado", response.json()["detail"])

    def test_predict_returns_sanitized_500_for_inference_failure(self):
        # Simula uma falha inesperada e verifica o log interno e a mensagem
        # genérica enviada ao cliente, sem revelar detalhes da exceção.
        with patch(
            "api.routers.predict_price",
            side_effect=RuntimeError("internal model details"),
        ), self.assertLogs("api.exception_handlers", level="ERROR"):
            client = TestClient(self.app, raise_server_exceptions=False)
            response = client.post("/predict", json=_valid_payload())

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": "Erro interno ao processar a requisição."},
        )
        self.assertNotIn("internal model details", response.text)

    def test_response_schema_rejects_invalid_output(self):
        # O contrato de saída também valida os dados gerados internamente:
        # preço positivo, versão não vazia e tipo numérico estrito.
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
