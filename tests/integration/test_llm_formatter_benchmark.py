import asyncio
import os

import pytest

from tests.integration.llm_formatter_benchmark.runner import RESULTS_PATH, run_benchmark


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VOCALANCE_RUN_LLM_BENCHMARK") != "1",
    reason="Set VOCALANCE_RUN_LLM_BENCHMARK=1 to run local GGUF benchmark (~40 LLM loads).",
)
def test_llm_formatter_benchmark_runs_and_writes_cache() -> None:
    data = asyncio.run(run_benchmark())
    assert RESULTS_PATH.is_file()
    assert data["completed_requests"] > 0
    assert data["runs"]
