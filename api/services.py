"""Carregamento dos artefatos do modelo e execução da inferência.

Esta camada concentra a lógica de Machine Learning, sem depender das rotas HTTP.
O modelo e o scaler são lidos durante a inicialização da aplicação e reutilizados
em memória pelas requisições de previsão.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from api.schemas import HealthResponse, PredictionRequest, PredictionResponse


# Os caminhos são relativos à raiz do projeto, não ao diretório de onde o
# servidor foi iniciado. Assim, a localização dos artefatos é previsível.
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "v1"
MODEL_FILE = MODEL_DIR / "modelo_gradientBoosting_v1.pkl"
SCALER_FILE = MODEL_DIR / "scaler_gradientBoosting_v1.pkl"
METRICS_FILE = MODEL_DIR / "metricas_v1.json"


@dataclass(frozen=True)
class ModelArtifacts:
    """Agrupa os artefatos e metadados necessários para fazer previsões.

    A instância é imutável para evitar que uma requisição altere acidentalmente
    referências compartilhadas por todas as outras requisições.
    """

    model: Any
    scaler: Any
    features: list[str]
    version: str


# Esta referência compartilhada é preenchida no startup pelo lifespan do FastAPI.
# Mantê-la vazia até a inicialização permite detectar uso incorreto do serviço.
_artifacts: ModelArtifacts | None = None


def _load_artifacts() -> ModelArtifacts:
    """Lê do disco e confere a compatibilidade dos artefatos no startup.

    Esta função é chamada pela inicialização da aplicação, nunca pela rota de
    previsão. Erros de arquivo ou incompatibilidade são levantados para impedir
    que a API comece a atender com um modelo inadequado.
    """
    # O JSON registra a versão do modelo e a lista ordenada de features usadas
    # no treinamento.
    with METRICS_FILE.open(encoding="utf-8") as metrics_file:
        metadata = json.load(metrics_file)

    # Primeiro valida o formato dos metadados, antes de usá-los para comparar
    # o contrato da API ou construir a estrutura de inferência.
    features = metadata.get("features_utilizadas")
    if not isinstance(features, list) or not all(
        isinstance(feature, str) for feature in features
    ):
        raise ValueError(f"Lista de features inválida em {METRICS_FILE}.")

    # A ordem também importa: o modelo foi treinado com as colunas em uma
    # sequência específica, portanto os nomes precisam coincidir exatamente.
    schema_features = list(PredictionRequest.model_fields)
    if features != schema_features:
        raise ValueError(
            "A ordem ou os nomes das features no schema não correspondem "
            "aos registrados em metricas_v1.json."
        )

    # Estes arquivos são carregados uma única vez aqui, no startup da aplicação.
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    # Verifica se os objetos serializados oferecem as operações necessárias
    # antes de guardá-los para uso pelas requisições.
    if not callable(getattr(model, "predict", None)):
        raise TypeError(f"O artefato em {MODEL_FILE} não oferece predict().")
    if not callable(getattr(scaler, "transform", None)):
        raise TypeError(f"O artefato em {SCALER_FILE} não oferece transform().")

    # Confere a dimensão esperada tanto no modelo quanto no scaler. Isso detecta
    # artefatos produzidos com uma quantidade diferente de colunas.
    expected_feature_count = len(features)
    for artifact_name, artifact in (("modelo", model), ("scaler", scaler)):
        artifact_feature_count = getattr(artifact, "n_features_in_", None)
        if artifact_feature_count != expected_feature_count:
            raise ValueError(
                f"O número de features do {artifact_name} ({artifact_feature_count}) "
                f"não corresponde às {expected_feature_count} features do metadata."
            )

    # Quando o scaler registra os nomes das colunas, valida também sua ordem.
    scaler_features = getattr(scaler, "feature_names_in_", None)
    if scaler_features is not None and list(scaler_features) != features:
        raise ValueError(
            "A ordem ou os nomes das features do scaler não correspondem "
            "aos registrados em metricas_v1.json."
        )

    # A versão será enviada ao cliente junto com cada previsão.
    version = metadata.get("versao")
    if not isinstance(version, str) or not version:
        raise ValueError(f"Versão inválida em {METRICS_FILE}.")

    return ModelArtifacts(
        model=model,
        scaler=scaler,
        features=features,
        version=version,
    )


def predict_price(request: PredictionRequest) -> PredictionResponse:
    """Transforma uma entrada validada em uma previsão usando artefatos em RAM."""
    # Falha claramente se o lifespan não tiver inicializado o serviço; não há
    # tentativa de carregar arquivos como alternativa durante uma requisição.
    artifacts = _artifacts
    if artifacts is None:
        raise RuntimeError(
            "Os artefatos do modelo não foram inicializados. "
            "A aplicação deve iniciar pelo lifespan do FastAPI."
        )

    # Converte os dados Pydantic em uma linha tabular e fixa a ordem das colunas
    # conforme os metadados verificados no startup.
    input_frame = pd.DataFrame(
        [request.model_dump()],
        columns=artifacts.features,
    )
    # A inferência deve receber os dados transformados pelo mesmo scaler usado
    # no treinamento; em seguida, o modelo estima o preço para essa única linha.
    scaled_features = artifacts.scaler.transform(input_frame)
    prediction = artifacts.model.predict(scaled_features)

    # A API espera exatamente um valor porque cada chamada representa um imóvel.
    if len(prediction) != 1:
        raise ValueError("O modelo deveria retornar exatamente uma previsão.")

    # Arredonda o resultado entregue pela API sem alterar a precisão interna do modelo.
    estimated_price = round(float(prediction[0]), 2)
    if not math.isfinite(estimated_price):
        raise ValueError("O modelo retornou uma previsão não finita.")

    return PredictionResponse(
        preco_estimado=estimated_price,
        versao_modelo=artifacts.version,
    )


def get_health_status() -> HealthResponse:
    """Verifica apenas as referências em memória; não acessa arquivos do modelo."""
    # A checagem confirma a presença dos objetos e das operações necessárias,
    # sem executar inferência nem acessar o disco.
    artifacts = _artifacts
    model_loaded = artifacts is not None and callable(
        getattr(artifacts.model, "predict", None)
    )
    scaler_loaded = artifacts is not None and callable(
        getattr(artifacts.scaler, "transform", None)
    )

    # Se esta função respondeu, o processo da API está ativo. O campo status
    # diferencia uma aplicação pronta para prever de outra sem artefatos.
    return HealthResponse(
        status="ok" if model_loaded and scaler_loaded else "degraded",
        service_active=True,
        model_loaded=model_loaded,
        scaler_loaded=scaler_loaded,
    )


def initialize_model() -> None:
    """Carrega os artefatos uma vez, antes de a aplicação aceitar requisições."""
    global _artifacts
    # Evita leituras repetidas caso a inicialização seja chamada novamente
    # durante o mesmo ciclo de vida da aplicação.
    if _artifacts is None:
        _artifacts = _load_artifacts()


def clear_model_cache() -> None:
    """Libera as referências em memória quando a aplicação é encerrada."""
    global _artifacts
    # Remove as referências globais para liberar os objetos quando o lifespan
    # encerra a aplicação.
    _artifacts = None
