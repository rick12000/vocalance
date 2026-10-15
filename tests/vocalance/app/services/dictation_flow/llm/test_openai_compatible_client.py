from unittest.mock import MagicMock, patch

import httpx

from vocalance.app.services.dictation_flow.llm.openai_compatible_client import chat_completions_sync


def _response(status: int, json_body: dict | None = None, text: str = "") -> httpx.Response:
    request = httpx.Request("POST", "http://127.0.0.1/v1/chat/completions")
    if json_body is not None:
        return httpx.Response(status, json=json_body, request=request)
    return httpx.Response(status, text=text, request=request)


def test_chat_completions_retries_without_stop_on_http_400():
    ok_payload = {"choices": [{"message": {"content": "Done."}}]}
    client = MagicMock()
    client.post.side_effect = [
        _response(400, text="bad stop"),
        _response(200, json_body=ok_payload),
    ]
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)

    with patch("vocalance.app.services.dictation_flow.llm.openai_compatible_client.httpx.Client", return_value=client):
        result = chat_completions_sync(
            "http://127.0.0.1:11434",
            "test",
            [{"role": "user", "content": "hi"}],
            32,
            10.0,
            stop=["\n\nHere is the"],
        )

    assert result == "Done."
    assert client.post.call_count == 2
    first_body = client.post.call_args_list[0].kwargs["json"]
    second_body = client.post.call_args_list[1].kwargs["json"]
    assert "stop" in first_body
    assert "stop" not in second_body
