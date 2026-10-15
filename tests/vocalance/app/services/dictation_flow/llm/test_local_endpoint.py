import pytest

from vocalance.app.services.dictation_flow.llm.local_endpoint import (
    normalize_chat_completions_url,
    validate_local_endpoint_base_url,
)


def test_normalize_chat_completions_url_appends_v1():
    assert (
        normalize_chat_completions_url("http://127.0.0.1:11434")
        == "http://127.0.0.1:11434/v1/chat/completions"
    )


def test_normalize_chat_completions_url_respects_existing_v1():
    assert (
        normalize_chat_completions_url("http://localhost:8080/v1")
        == "http://localhost:8080/v1/chat/completions"
    )


def test_validate_accepts_loopback_hosts():
    assert validate_local_endpoint_base_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert validate_local_endpoint_base_url("http://localhost:8080/") == "http://localhost:8080"


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com",
        "http://192.168.1.10:11434",
        "http://user:pass@127.0.0.1:11434",
    ],
)
def test_validate_rejects_non_loopback_or_credentials(url: str):
    with pytest.raises(ValueError):
        validate_local_endpoint_base_url(url)
