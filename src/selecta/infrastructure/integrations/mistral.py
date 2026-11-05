"""Mistral chat and embedding models over the HTTP API."""

import json
import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from selecta.application.conversation import Message, Reply, ToolCall, ToolSpec
from selecta.application.ports import ModelError, ModelOutputError
from selecta.infrastructure import config

logger = logging.getLogger(__name__)


class MistralError(ModelError):
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
    """ChatModel and AgentModel on Mistral's chat completions."""

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

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        """Return the next reply."""
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": config.MISTRAL.AGENT_TEMPERATURE,
            "max_tokens": config.MISTRAL.MAX_OUTPUT_TOKENS,
            "messages": [{"role": "system", "content": system}, *map(_to_wire, messages)],
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t.model_dump()} for t in tools]
        data = _post(self._http, "/chat/completions", body)
        try:
            message = data["choices"][0]["message"]
            calls = [
                ToolCall(
                    id=c["id"],
                    name=c["function"]["name"],
                    arguments=json.loads(c["function"]["arguments"] or "{}"),
                )
                for c in message.get("tool_calls") or []
            ]
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as e:
            raise ModelOutputError(f"{self.model} returned a malformed reply") from e
        usage = data.get("usage") or {}
        return Reply(
            text=message.get("content") or None,
            tool_calls=calls,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


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


def _to_wire(message: Message) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role, "content": message.content or ""}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in message.tool_calls
        ]
    if message.role == "tool":
        wire["tool_call_id"] = message.tool_call_id
        wire["name"] = message.name
    return wire
