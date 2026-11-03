"""Rotas HTTP da API de previsão imobiliária.

Este módulo conecta as requisições recebidas pelo FastAPI às funções do serviço.
As regras de carregamento do modelo e de inferência ficam em ``api.services``;
os contratos dos dados recebidos e devolvidos ficam em ``api.schemas``.
"""

from fastapi import APIRouter

from api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from api.services import get_health_status, predict_price


# Agrupa os endpoints da API para que possam ser registrados juntos em main.py.
# A tag aparece como categoria na documentação automática do Swagger.
router = APIRouter(tags=["Previsões"])


# O response_model valida e documenta o formato devolvido pelo endpoint.
# A tag específica separa a checagem de saúde das rotas de previsão no Swagger.
@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoramento"],
    summary="Verificar estado da API e do modelo",
)
def health_check() -> HealthResponse:
    """Informa se a API está ativa e se modelo e scaler estão disponíveis."""
    # Consulta o estado que o serviço manteve em memória durante o startup.
    # Não acessa o disco nem tenta carregar novamente os artefatos do modelo.
    return get_health_status()


# A rota declara os códigos de erro esperados para que também apareçam no Swagger.
# O FastAPI/Pydantic trata automaticamente entradas inválidas com HTTP 422;
# o handler da aplicação trata JSON malformado (400) e falhas inesperadas (500).
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
    """Encaminha uma entrada validada ao serviço e devolve a previsão."""
    # Antes de chamar esta função, o FastAPI valida o JSON usando
    # PredictionRequest. Se os campos ou valores forem inválidos, a execução
    # para aqui e o cliente recebe HTTP 422, sem enviar dados ao modelo.
    #
    # A rota delega o cálculo ao serviço: não carrega arquivos nem acessa
    # diretamente o modelo. O serviço reutiliza os artefatos carregados no startup.
    return predict_price(request)
