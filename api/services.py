"""Carregamento dos artefatos do modelo e execução da inferência."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from api.schemas import HealthResponse, PredictionRequest, PredictionResponse


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "v1"
MODEL_FILE = MODEL_DIR / "modelo_gradientBoosting_v1.pkl"
SCALER_FILE = MODEL_DIR / "scaler_gradientBoosting_v1.pkl"
METRICS_FILE = MODEL_DIR / "metricas_v1.json"


@dataclass(frozen=True)
class ModelArtifacts:
    """Artefatos mantidos em memória enquanto a aplicação está ativa."""

    model: Any
    scaler: Any
    features: list[str]
    version: str


_artifacts: ModelArtifacts | None = None


def _load_artifacts() -> ModelArtifacts:
    """Lê do disco e valida os artefatos durante o startup da aplicação."""
    with METRICS_FILE.open(encoding="utf-8") as metrics_file:
        metadata = json.load(metrics_file)

    features = metadata.get("features_utilizadas")
    if not isinstance(features, list) or not all(
        isinstance(feature, str) for feature in features
    ):
        raise ValueError(f"Lista de features inválida em {METRICS_FILE}.")

    schema_features = list(PredictionRequest.model_fields)
    if features != schema_features:
        raise ValueError(
            "A ordem ou os nomes das features no schema não correspondem "
            "aos registrados em metricas_v1.json."
        )

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    if not callable(getattr(model, "predict", None)):
        raise TypeError(f"O artefato em {MODEL_FILE} não oferece predict().")
    if not callable(getattr(scaler, "transform", None)):
        raise TypeError(f"O artefato em {SCALER_FILE} não oferece transform().")

    expected_feature_count = len(features)
    for artifact_name, artifact in (("modelo", model), ("scaler", scaler)):
        artifact_feature_count = getattr(artifact, "n_features_in_", None)
        if artifact_feature_count != expected_feature_count:
            raise ValueError(
                f"O número de features do {artifact_name} ({artifact_feature_count}) "
                f"não corresponde às {expected_feature_count} features do metadata."
            )

    scaler_features = getattr(scaler, "feature_names_in_", None)
    if scaler_features is not None and list(scaler_features) != features:
        raise ValueError(
            "A ordem ou os nomes das features do scaler não correspondem "
            "aos registrados em metricas_v1.json."
        )

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
    """Infere usando somente os artefatos já carregados na memória."""
    artifacts = _artifacts
    if artifacts is None:
        raise RuntimeError(
            "Os artefatos do modelo não foram inicializados. "
            "A aplicação deve iniciar pelo lifespan do FastAPI."
        )

    input_frame = pd.DataFrame(
        [request.model_dump()],
        columns=artifacts.features,
    )
    scaled_features = artifacts.scaler.transform(input_frame)
    prediction = artifacts.model.predict(scaled_features)

    if len(prediction) != 1:
        raise ValueError("O modelo deveria retornar exatamente uma previsão.")

    estimated_price = float(prediction[0])
    if not math.isfinite(estimated_price):
        raise ValueError("O modelo retornou uma previsão não finita.")

    return PredictionResponse(
        preco_estimado=estimated_price,
        versao_modelo=artifacts.version,
    )


def get_health_status() -> HealthResponse:
    """Verifica apenas as referências em memória; não acessa arquivos do modelo."""
    artifacts = _artifacts
    model_loaded = artifacts is not None and callable(
        getattr(artifacts.model, "predict", None)
    )
    scaler_loaded = artifacts is not None and callable(
        getattr(artifacts.scaler, "transform", None)
    )

    # O endpoint foi alcançado, então a API está ativa; status distingue artefatos ausentes.
    return HealthResponse(
        status="ok" if model_loaded and scaler_loaded else "degraded",
        service_active=True,
        model_loaded=model_loaded,
        scaler_loaded=scaler_loaded,
    )


def initialize_model() -> None:
    """Carrega os artefatos uma vez, antes de a aplicação aceitar requisições."""
    global _artifacts
    if _artifacts is None:
        _artifacts = _load_artifacts()


def clear_model_cache() -> None:
    """Libera as referências em memória quando a aplicação é encerrada."""
    global _artifacts
    _artifacts = None
