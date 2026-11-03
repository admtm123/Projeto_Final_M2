"""Ponto de entrada e composição da aplicação FastAPI.

Este módulo conecta o ciclo de vida da aplicação, o tratamento de exceções
e as rotas HTTP. A lógica de previsão permanece na camada de serviços.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.exception_handlers import register_exception_handlers
from api.routers import router
from api.services import clear_model_cache, initialize_model


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Prepara os recursos no startup e os libera no encerramento da aplicação.

    O lifespan roda uma vez por ciclo de vida do processo: o trecho antes de
    ``yield`` prepara a aplicação, e o trecho depois dele executa a limpeza.
    """
    try:
        # Inicializa modelo e scaler antes de o servidor começar a processar
        # requisições. As rotas reutilizam esses objetos mantidos em memória.
        initialize_model()
        # O FastAPI atende as requisições enquanto a execução está suspensa aqui.
        yield
    finally:
        # Executa a limpeza ao encerrar, inclusive se ocorrer uma exceção durante
        # o ciclo de vida, liberando as referências aos artefatos carregados.
        clear_model_cache()


# Instância principal usada pelo servidor ASGI (por exemplo, ``uvicorn main:app``).
# O lifespan vincula o carregamento do modelo ao ciclo de vida desta aplicação.
app = FastAPI(
    title="API de Previsão de Preços de Imóveis",
    version="1.0.0",
    lifespan=lifespan,
)

# Registra respostas padronizadas para erros HTTP e de validação da aplicação.
register_exception_handlers(app)

# Adiciona à aplicação os endpoints definidos em api.routers, mantendo a
# declaração das rotas separada da configuração global do FastAPI.
app.include_router(router)
