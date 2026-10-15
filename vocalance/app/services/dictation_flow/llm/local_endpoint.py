from ipaddress import ip_address
from typing import Literal
from urllib.parse import urlparse

LLMProviderKind = Literal["shipped", "local_endpoint"]

_LOCAL_HOSTNAMES = frozenset({"localhost"})


def normalize_chat_completions_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def validate_local_endpoint_base_url(base_url: str) -> str:
    raw = base_url.strip()
    if not raw:
        raise ValueError("Endpoint base URL is required.")
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Endpoint URL must use http or https.")
    if parsed.username or parsed.password:
        raise ValueError("Endpoint URL must not include credentials.")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("Endpoint URL must include a host.")
    if host in _LOCAL_HOSTNAMES:
        return raw.rstrip("/")
    try:
        addr = ip_address(host)
    except ValueError:
        raise ValueError("Endpoint host must be localhost, 127.0.0.1, or ::1.")
    if not addr.is_loopback:
        raise ValueError("Endpoint host must be a loopback address.")
    return raw.rstrip("/")
