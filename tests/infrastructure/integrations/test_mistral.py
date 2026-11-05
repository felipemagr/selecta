import json

import httpx
import pytest

from selecta.application.ports import ModelOutputError
from selecta.domain.enrichment import Enrichment
from selecta.infrastructure import config
from selecta.infrastructure.integrations import mistral
from selecta.infrastructure.integrations.mistral import MistralChat


def _chat(*responses: httpx.Response) -> MistralChat:
    queue = list(responses)
    return MistralChat(mistral.connect("key", httpx.MockTransport(lambda _: queue.pop(0))))


def _answer(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@pytest.fixture(autouse=True)
def no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.MISTRAL, "RETRY_WAIT_SECONDS", 0)


def test_rate_limit_is_retried() -> None:
    answer = json.dumps({"mood": "calm", "energy": "low"})
    chat = _chat(httpx.Response(429), _answer(answer))

    assert chat.complete_json("s", "u", Enrichment).mood == "calm"


def test_answer_outside_schema_raises() -> None:
    chat = _chat(_answer(json.dumps({"mood": "sleepy", "energy": "low"})))

    with pytest.raises(ModelOutputError):
        chat.complete_json("s", "u", Enrichment)
