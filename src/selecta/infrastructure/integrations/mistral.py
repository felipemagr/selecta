"""Mistral chat and embedding models over the HTTP API."""

import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from selecta.application.ports import ModelOutputError
from selecta.infrastructure import config

logger = logging.getLogger(__name__)


class MistralError(Exception):
    """The Mistral API failed in a way a retry did not fix."""


def connect(api_key: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Return an HTTP client authenticated against the Mistral API."""
    return httpx.Client(
        base_url=config.MISTRAL.API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=config.MISTRAL.TIMEOUT_SECONDS,
        transport=transport,
    )


class MistralChat:
    """ChatModel on Mistral's chat completions with strict JSON schema output."""

    def __init__(self, http: httpx.Client, model: str = config.MISTRAL.CHAT_MODEL) -> None:
        self._http = http
        self.model = model

    def complete_json[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        """Return the model's answer to `user`, validated against `schema`."""
        json_schema = {**schema.model_json_schema(), "additionalProperties": False}
        data = _post(
            self._http,
            "/chat/completions",
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": config.MISTRAL.MAX_OUTPUT_TOKENS,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": schema.__name__, "schema": json_schema, "strict": True},
                },
            },
        )
        try:
            return schema.model_validate_json(data["choices"][0]["message"]["content"])
        except ValidationError as e:
            raise ModelOutputError(f"{self.model} returned an invalid {schema.__name__}") from e


class MistralEmbedder:
    """Embedder on Mistral's embeddings endpoint."""

    def __init__(self, http: httpx.Client, model: str = config.MISTRAL.EMBED_MODEL) -> None:
        self._http = http
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text, in order."""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), config.MISTRAL.EMBED_BATCH_SIZE):
            batch = texts[start : start + config.MISTRAL.EMBED_BATCH_SIZE]
            data = _post(self._http, "/embeddings", {"model": self.model, "input": batch})
            embeddings += [item["embedding"] for item in data["data"]]
        return embeddings


def _post(http: httpx.Client, path: str, body: dict[str, Any]) -> dict[str, Any]:
    for attempt in range(config.MISTRAL.MAX_RETRIES):
        try:
            response = http.post(path, json=body)
        except httpx.TransportError as e:
            logger.warning("%s on %s, retrying", type(e).__name__, path)
        else:
            if response.status_code != 429 and response.status_code < 500:
                if response.is_error:
                    raise MistralError(
                        f"POST {path} failed: {response.status_code} {response.text}"
                    )
                return response.json()
            logger.warning("%s on %s, retrying", response.status_code, path)
        if attempt < config.MISTRAL.MAX_RETRIES - 1:
            time.sleep(config.MISTRAL.RETRY_WAIT_SECONDS * 2**attempt)
    raise MistralError(f"POST {path} failed after {config.MISTRAL.MAX_RETRIES} tries")
