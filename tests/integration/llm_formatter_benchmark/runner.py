"""Run local LLM formatter benchmark; writes cache/llm_formatter_benchmark/results.json."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tests.integration.llm_formatter_benchmark.cases import (
    AMEND_CASES,
    BENCHMARK_MODEL_IDS,
    REPS_PER_CASE,
    SMART_DICTATION_CASES,
)
from tests.integration.llm_formatter_benchmark.scoring import score_amend, score_smart
from vocalance.app.config.app_config import GlobalAppConfig, get_whitelisted_llm_model
from vocalance.app.event_bus import EventBus
from vocalance.app.services.dictation_flow.llm.agentic_prompt_service import AgenticPromptService
from vocalance.app.services.dictation_flow.llm.llm_service import LLMService
from vocalance.app.services.storage.runtime_configuration import RuntimeConfigurationStore
from vocalance.app.services.storage.storage_service import StorageService

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "cache" / "llm_formatter_benchmark"
RESULTS_PATH = OUTPUT_DIR / "results.json"


def _aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_key: Dict[str, Dict[str, List[bool]]] = {}
    for run in runs:
        if run.get("error"):
            continue
        model = run["model_id"]
        route = run["route"]
        key = f"{model}|{route}"
        by_key.setdefault(key, {})
        for criterion, passed in run.get("scores", {}).items():
            by_key[key].setdefault(criterion, []).append(bool(passed))

    summary: Dict[str, Any] = {}
    for key, criteria in by_key.items():
        model_id, route = key.split("|", 1)
        summary.setdefault(model_id, {}).setdefault(route, {})
        for criterion, values in criteria.items():
            rate = sum(values) / len(values) if values else 0.0
            summary[model_id][route][criterion] = round(rate, 3)
        all_pass = []
        for run in runs:
            if run.get("error") or run["model_id"] != model_id or run["route"] != route:
                continue
            all_pass.append(all(run["scores"].values()))
        summary[model_id][route]["all_criteria_pass"] = round(sum(all_pass) / len(all_pass) if all_pass else 0.0, 3)
    return summary


async def _bootstrap() -> tuple[GlobalAppConfig, AgenticPromptService, LLMService]:
    config = GlobalAppConfig()
    event_bus = EventBus()
    storage = StorageService(config=config)
    runtime = RuntimeConfigurationStore(event_bus=event_bus, config=config, storage=storage)
    await runtime.initialize()
    agentic = AgenticPromptService(event_bus=event_bus, config=config, storage=storage)
    await agentic.initialize()
    llm = LLMService(event_bus=event_bus, config=config)
    return config, agentic, llm


async def run_benchmark(
    model_ids: Optional[List[str]] = None,
    reps: int = REPS_PER_CASE,
) -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config, agentic, llm = await _bootstrap()
    prompt = agentic.get_prompt_for_inference()
    models = list(model_ids or BENCHMARK_MODEL_IDS)

    runs: List[Dict[str, Any]] = []
    skipped_models: List[str] = []

    for model_id in models:
        spec = get_whitelisted_llm_model(model_id)
        if spec is None:
            skipped_models.append(model_id)
            continue
        if not llm.is_whitelisted_bundle_on_disk(model_id):
            skipped_models.append(model_id)
            continue

        config.llm.selected_model_id = model_id

        for case in SMART_DICTATION_CASES:
            for rep in range(1, reps + 1):
                started = time.perf_counter()
                err: Optional[str] = None
                output = ""
                try:
                    result = await llm.process_dictation(case.transcript, prompt)
                    output = (result or "").strip()
                except Exception as exc:
                    err = str(exc)
                duration_ms = int((time.perf_counter() - started) * 1000)
                scores = score_smart(case, case.transcript, output) if not err else {}
                runs.append(
                    {
                        "model_id": model_id,
                        "route": "smart",
                        "case_id": case.id,
                        "rep": rep,
                        "input": {"transcript": case.transcript},
                        "output": output,
                        "scores": scores,
                        "error": err,
                        "duration_ms": duration_ms,
                    }
                )

        for case in AMEND_CASES:
            for rep in range(1, reps + 1):
                started = time.perf_counter()
                err = None
                output = ""
                try:
                    result = await llm.process_amend_streaming(
                        case.clipboard,
                        case.spoken,
                        prompt,
                        stream_session_id=None,
                    )
                    output = (result or "").strip()
                except Exception as exc:
                    err = str(exc)
                duration_ms = int((time.perf_counter() - started) * 1000)
                scores = score_amend(case, output) if not err else {}
                runs.append(
                    {
                        "model_id": model_id,
                        "route": "amend",
                        "case_id": case.id,
                        "rep": rep,
                        "input": {"clipboard": case.clipboard, "spoken": case.spoken},
                        "output": output,
                        "scores": scores,
                        "error": err,
                        "duration_ms": duration_ms,
                    }
                )

    payload: Dict[str, Any] = {
        "budget": {
            "models": models,
            "reps_per_case": reps,
            "smart_cases": len(SMART_DICTATION_CASES),
            "amend_cases": len(AMEND_CASES),
            "expected_requests": len(models) * reps * (len(SMART_DICTATION_CASES) + len(AMEND_CASES)),
        },
        "skipped_models": skipped_models,
        "agentic_prompt_chars": len(prompt),
        "runs": runs,
        "aggregate": _aggregate(runs),
    }
    payload["completed_requests"] = len([r for r in runs if not r.get("error")])

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    await agentic.shutdown()
    return payload


def load_results() -> Dict[str, Any]:
    with RESULTS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def print_report(data: Optional[Dict[str, Any]] = None) -> None:
    data = data or load_results()
    print(f"Results file: {RESULTS_PATH}")
    print(f"Completed: {data.get('completed_requests')} / {len(data.get('runs', []))} runs")
    if data.get("skipped_models"):
        print(f"Skipped models (missing on disk): {data['skipped_models']}")
    agg = data.get("aggregate", {})
    for model_id, routes in agg.items():
        print(f"\n=== {model_id} ===")
        for route, criteria in routes.items():
            print(f"  [{route}]")
            for name, rate in sorted(criteria.items()):
                print(f"    {name}: {rate * 100:.1f}%")


def main() -> None:
    data = asyncio.run(run_benchmark())
    print_report(data)


if __name__ == "__main__":
    main()
