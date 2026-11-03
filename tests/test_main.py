"""Testes de integração da composição e do ciclo de vida da aplicação."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


class ApplicationLifecycleTests(unittest.TestCase):
    """Verifica que o ponto de entrada prepara e limpa os recursos da API."""

    def test_lifespan_loads_and_clears_model_artifacts(self):
        # Substitui o carregamento real para testar somente a integração do
        # lifespan, sem depender de arquivos serializados ou do modelo.
        with (
            patch("main.initialize_model") as initialize_model,
            patch("main.clear_model_cache") as clear_model_cache,
            TestClient(app) as client,
        ):
            # Entrar no contexto do TestClient inicia a aplicação e sair dele
            # encerra o lifespan. A chamada também confirma que a app responde.
            response = client.get("/openapi.json")
            self.assertEqual(response.status_code, 200)
            self.assertIn("/predict", response.json()["paths"])

        # O startup precisa inicializar os artefatos, e o shutdown deve limpá-los.
        initialize_model.assert_called_once_with()
        clear_model_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
