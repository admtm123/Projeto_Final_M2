"""Ponto de entrada da API FastAPI."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers import router
from api.services import clear_model_cache, initialize_model


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Inicializa os artefatos do modelo e limpa o cache ao encerrar."""
    try:
        initialize_model()
        yield
    finally:
        clear_model_cache()


app = FastAPI(
    title="API de Previsão de Preços de Imóveis",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
