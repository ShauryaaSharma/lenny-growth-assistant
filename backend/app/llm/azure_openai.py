"""Azure OpenAI.

The same chat-completions protocol as `openai_compat`, so the request and
response handling is inherited unchanged. Three things differ, and are all
this class sets:

- the URL names a *deployment*, not a model:
  {endpoint}/openai/deployments/{deployment}/chat/completions
- every request carries an `api-version` query parameter;
- the key goes in an `api-key` header, not `Authorization: Bearer`.

The badge shows the deployment name, since that is what Azure routes on; which
model sits behind it is configured in the Azure portal.
"""

from __future__ import annotations

from urllib.parse import quote

import httpx

from app.config import get_settings
from app.llm.openai_compat import OpenAICompatProvider


class AzureOpenAIProvider(OpenAICompatProvider):
    name = "azure_openai"
    models_path = "/models"
    key_variable = "AZURE_OPENAI_API_KEY"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        settings = get_settings()
        self.base_url = f"{settings.azure_openai_endpoint.rstrip('/')}/openai"
        self._model = settings.azure_openai_deployment
        self.api_key = settings.azure_openai_api_key
        self.timeout = settings.llm_timeout_seconds
        self.chat_path = f"/deployments/{quote(self._model, safe='')}/chat/completions"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            params={"api-version": settings.azure_openai_api_version},
            headers={"api-key": self.api_key} if self.api_key else {},
            transport=transport,
        )
