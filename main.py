"""Ponto de entrada da API FastAPI."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.exception_handlers import register_exception_handlers
from api.routers import router
from api.services import clear_model_cache, initialize_model


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Carrega os artefatos antes das requisições e libera-os ao encerrar."""
    try:
        # Carrega o modelo e o scaler em memória antes de aceitar requisições.
        initialize_model()
        yield
    finally:
        # Descarta as referências em memória mesmo se o app encerrar com erro.
        clear_model_cache()


app = FastAPI(
    title="API de Previsão de Preços de Imóveis",
    version="1.0.0",
    lifespan=lifespan,
)
register_exception_handlers(app)
app.include_router(router)
