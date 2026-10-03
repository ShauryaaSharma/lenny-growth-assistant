"""Error responses as the OpenAPI schema should describe them.

Every error leaves through one envelope (see the handlers in `app.main`),
but the schema never said so: FastAPI documented each 422 as its own
default `{"detail": [...]}`, and no 404 or model-provider error at all.
A client generated from `/openapi.json` would parse every error wrongly.

Declaring 422 on a route replaces FastAPI's default for that route, so
contract tests (Schemathesis, in CI) now check real error bodies against
the real envelope.
"""

from __future__ import annotations

from typing import Any

from app.schemas.api import ErrorResponse, ValidationErrorResponse

INVALID: dict[int | str, dict[str, Any]] = {
    422: {"model": ValidationErrorResponse, "description": "The request failed validation"},
}

NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "No such session or artifact"},
}

MODEL_PROVIDER: dict[int | str, dict[str, Any]] = {
    502: {"model": ErrorResponse, "description": "The model provider rejected or garbled the call"},
    503: {"model": ErrorResponse, "description": "The model provider is unreachable"},
    504: {"model": ErrorResponse, "description": "The model provider timed out"},
}
