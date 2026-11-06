import json

import httpx
import pytest

from selecta.application.conversation import Message
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


def test_tool_calls_are_parsed() -> None:
    call = {"id": "c1", "function": {"name": "get_release", "arguments": '{"release_id": 7}'}}
    response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "", "tool_calls": [call]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    )

    reply = _chat(response).chat("s", [Message(role="user", content="hi")], [])

    assert reply.text is None
    assert [(c.name, c.arguments) for c in reply.tool_calls] == [("get_release", {"release_id": 7})]


def _stream(*chunks: dict[str, object]) -> str:
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    return "\n\n".join([*lines, "data: [DONE]"]) + "\n\n"


def _calls(*calls: dict[str, object]) -> dict[str, object]:
    return {"choices": [{"delta": {"tool_calls": list(calls)}}]}


def test_streamed_reply_reports_text_and_tool_calls_as_they_arrive() -> None:
    brief = {"name": "update_brief", "arguments": '{"mood": '}
    body = _stream(
        {"choices": [{"delta": {"content": "Calm it "}}]},
        {"choices": [{"delta": {"content": "is."}}]},
        _calls(
            {"index": 0, "id": "c1", "function": brief},
            {"index": 1, "id": "c2", "function": {"name": "skip_questions", "arguments": "{}"}},
        ),
        _calls({"index": 0, "function": {"arguments": '"calm"}'}}),
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
    )
    chat = _chat(httpx.Response(429), httpx.Response(200, text=body))
    texts: list[str] = []

    reply = chat.chat("s", [Message(role="user", content="hi")], [], texts.append)

    assert texts == ["Calm it ", "is."]
    assert reply.text == "Calm it is." and reply.tool_calls[0].arguments == {"mood": "calm"}
    assert [c.name for c in reply.tool_calls] == ["update_brief", "skip_questions"]
    assert (reply.input_tokens, reply.output_tokens) == (7, 3)
