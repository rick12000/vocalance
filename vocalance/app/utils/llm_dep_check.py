import importlib.util


def llm_deps_available() -> bool:
    """Return True if the optional LLM dependencies (llama_cpp, huggingface_hub) are installed."""
    return importlib.util.find_spec("llama_cpp") is not None
