"""Tratamento centralizado de erros HTTP da API."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


async def handle_request_validation_error(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Separa JSON malformado (400) de dados que violam o contrato (422)."""
    errors = exc.errors()
    if any(error.get("type") == "json_invalid" for error in errors):
        return JSONResponse(
            status_code=400,
            content={"detail": "O corpo da requisição contém JSON malformado."},
        )

    # Não devolve valores de entrada no erro, evitando expor conteúdo enviado pelo cliente.
    safe_errors = [
        {
            "loc": error.get("loc", ()),
            "msg": error.get("msg", "Valor inválido."),
            "type": error.get("type", "value_error"),
        }
        for error in errors
    ]
    return JSONResponse(status_code=422, content={"detail": safe_errors})


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Registra detalhes internos e devolve uma resposta 500 sem expor a exceção."""
    logger.exception(
        "Erro interno ao processar %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno ao processar a requisição."},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Registra handlers comuns para erros de validação e falhas inesperadas."""
    app.add_exception_handler(
        RequestValidationError,
        handle_request_validation_error,
    )
    app.add_exception_handler(Exception, handle_unexpected_error)
