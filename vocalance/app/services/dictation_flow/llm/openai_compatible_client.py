import json
from typing import Any, Dict, List, Optional

import httpx

from vocalance.app.services.dictation_flow.llm.local_endpoint import normalize_chat_completions_url


class OpenAICompatibleChatError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def extract_assistant_content(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenAICompatibleChatError("Response missing choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise OpenAICompatibleChatError("Response missing message.")
    content = message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise OpenAICompatibleChatError("Unexpected message content type.")
    return content


def _post_chat_completion(
    client: httpx.Client,
    url: str,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    response = client.post(url, json=body)
    if response.status_code >= 400:
        detail = response.text[:500]
        raise OpenAICompatibleChatError(
            f"HTTP {response.status_code}: {detail}",
            status_code=response.status_code,
        )
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise OpenAICompatibleChatError("Response was not JSON.") from exc
    if not isinstance(payload, dict):
        raise OpenAICompatibleChatError("Response JSON was not an object.")
    return payload


def chat_completions_sync(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    timeout_sec: float,
    stop: Optional[List[str]] = None,
) -> str:
    url = normalize_chat_completions_url(base_url)
    base_body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.7,
    }
    with httpx.Client(timeout=timeout_sec) as client:
        if stop:
            body_with_stop = {**base_body, "stop": stop}
            try:
                payload = _post_chat_completion(client, url, body_with_stop)
            except OpenAICompatibleChatError as exc:
                if exc.status_code != 400:
                    raise
                payload = _post_chat_completion(client, url, base_body)
        else:
            payload = _post_chat_completion(client, url, base_body)
    return extract_assistant_content(payload)
