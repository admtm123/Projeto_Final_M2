"""Rotas HTTP da API de previsão imobiliária."""

from fastapi import APIRouter

from api.schemas import PredictionRequest, PredictionResponse
from api.services import predict_price


router = APIRouter(tags=["Previsões"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prever preço de imóvel",
)
def create_prediction(request: PredictionRequest) -> PredictionResponse:
    """Valida os dados recebidos e solicita a previsão ao serviço."""
    return predict_price(request)
