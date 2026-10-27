"""Rotas HTTP da API de previsão imobiliária."""

from fastapi import APIRouter

from api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from api.services import get_health_status, predict_price


router = APIRouter(tags=["Previsões"])


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoramento"],
    summary="Verificar estado da API e do modelo",
)
def health_check() -> HealthResponse:
    """Responde sempre HTTP 200 e informa se o modelo está disponível em RAM."""
    # A checagem consulta somente o estado mantido em memória; nunca carrega arquivos.
    return get_health_status()


@router.post(
    "/predict",
    status_code=200,
    response_model=PredictionResponse,
    summary="Prever preço de imóvel",
    responses={
        400: {"description": "JSON da requisição malformado."},
        422: {"description": "Dados que não atendem ao schema de entrada."},
        500: {"description": "Erro interno durante o processamento da previsão."},
    },
)
def create_prediction(request: PredictionRequest) -> PredictionResponse:
    """Recebe a entrada validada pelo Pydantic e retorna a previsão do serviço."""
    # FastAPI valida o corpo antes desta função; a inferência usa artefatos em RAM,
    # já carregados no startup, sem reler arquivos a cada requisição.
    return predict_price(request)
