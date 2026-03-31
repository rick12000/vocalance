"""Whitelisted local LLM artifacts from official Qwen GGUF repositories (Q5_K_M)."""

from dataclasses import dataclass
from typing import Dict, Final, Optional, Tuple

# Official Qwen GGUF hubs: https://huggingface.co/Qwen/Qwen2.5-*-Instruct-GGUF


@dataclass(frozen=True)
class WhitelistedLLMModel:
    """One selectable model: Hugging Face repo, GGUF file name(s), and UI metadata."""

    id: str
    label: str
    repo_id: str
    gguf_filenames: Tuple[str, ...]
    model_card_url: str

    @property
    def load_path_filename(self) -> str:
        """Filename passed to llama.cpp (first shard for split GGUF)."""
        return self.gguf_filenames[0]


WHITELISTED_LLM_MODELS: Final[Tuple[WhitelistedLLMModel, ...]] = (
    WhitelistedLLMModel(
        id="qwen2.5-1.5b-q5km",
        label="Qwen 2.5 1.5B Instruct (Q5_K_M, CPU)",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        gguf_filenames=("qwen2.5-1.5b-instruct-q5_k_m.gguf",),
        model_card_url="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    ),
    WhitelistedLLMModel(
        id="qwen2.5-3b-q5km",
        label="Qwen 2.5 3B Instruct (Q5_K_M, CPU)",
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        gguf_filenames=("qwen2.5-3b-instruct-q5_k_m.gguf",),
        model_card_url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF",
    ),
    WhitelistedLLMModel(
        id="qwen2.5-7b-q5km",
        label="Qwen 2.5 7B Instruct (Q5_K_M, CPU)",
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        gguf_filenames=(
            "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
            "qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf",
        ),
        model_card_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF",
    ),
)

DEFAULT_LLM_MODEL_ID: Final[str] = "qwen2.5-1.5b-q5km"

_LLM_BY_ID: Dict[str, WhitelistedLLMModel] = {m.id: m for m in WHITELISTED_LLM_MODELS}


def get_whitelisted_llm_model(model_id: str) -> Optional[WhitelistedLLMModel]:
    return _LLM_BY_ID.get(model_id)


def is_whitelisted_llm_model_id(model_id: str) -> bool:
    return model_id in _LLM_BY_ID


def all_whitelisted_llm_model_ids() -> Tuple[str, ...]:
    return tuple(m.id for m in WHITELISTED_LLM_MODELS)
