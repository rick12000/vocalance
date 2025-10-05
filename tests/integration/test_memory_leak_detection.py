"""
Memory leak detection tests using psutil and tracemalloc.
"""
import asyncio
import gc
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import psutil
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iris.config.app_config import GlobalAppConfig
from iris.event_bus import EventBus


class MemoryProfiler:
    def __init__(self, component_name: str = "test"):
        self.component_name = component_name
        self.process = psutil.Process(os.getpid())
        self.snapshots: List[Dict[str, Any]] = []
        self.tracemalloc_enabled = False
        self.warning_threshold_mb = 50.0
        self.error_threshold_mb = 100.0

    def start_profiling(self):
        tracemalloc.start()
        self.tracemalloc_enabled = True
        self.take_snapshot("baseline")

    def take_snapshot(self, label: str = "snapshot") -> Dict[str, Any]:
        memory_info = self.process.memory_info()
        memory_full = self.process.memory_full_info()

        snapshot = {
            'label': label,
            'memory': {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'uss_mb': memory_full.uss / (1024 * 1024),
                'timestamp': time.time()
            },
            'gc_count': gc.get_count()
        }

        if self.tracemalloc_enabled:
            current, peak = tracemalloc.get_traced_memory()
            snapshot['tracemalloc'] = {'current_mb': current / (1024 * 1024)}

        self.snapshots.append(snapshot)
        return snapshot
        
    def analyze_growth(self) -> Dict[str, Any]:
        if len(self.snapshots) < 2:
            return {'error': 'No snapshots'}

        baseline = self.snapshots[0]['memory']
        current = self.snapshots[-1]['memory']
        uss_values = [s['memory']['uss_mb'] for s in self.snapshots]

        return {
            'component': self.component_name,
            'uss_growth_mb': current['uss_mb'] - baseline['uss_mb'],
            'peak_uss_mb': max(uss_values),
            'snapshot_count': len(self.snapshots),
            'baseline_uss_mb': baseline['uss_mb'],
            'current_uss_mb': current['uss_mb']
        }
        
    def detect_leak(self) -> Tuple[bool, str, Dict[str, Any]]:
        analysis = self.analyze_growth()
        if 'error' in analysis:
            return False, "no_data", analysis

        growth = analysis['uss_growth_mb']
        if growth > self.error_threshold_mb:
            return True, "severe", analysis
        if growth > self.warning_threshold_mb:
            return True, "moderate", analysis

        return False, "ok", analysis
        
    def stop_profiling(self) -> Dict[str, Any]:
        if self.tracemalloc_enabled:
            tracemalloc.stop()
            self.tracemalloc_enabled = False
        return self.analyze_growth()

    def print_report(self):
        analysis = self.analyze_growth()
        has_leak, severity, _ = self.detect_leak()

        print(f"\n{self.component_name}: {severity.upper()} ({analysis.get('uss_growth_mb', 0):.1f} MB growth)")
        if has_leak:
            print("WARNING: Memory leak detected!")


async def run_memory_test(profiler, component_name, test_func, max_growth_mb=50.0):
    """Run a memory test with common pattern."""
    profiler.component_name = component_name
    profiler.warning_threshold_mb = max_growth_mb
    profiler.error_threshold_mb = max_growth_mb * 2

    profiler.take_snapshot("start")

    await test_func(profiler)

    profiler.take_snapshot("end")

    has_leak, severity, details = profiler.detect_leak()
    profiler.print_report()

    assert not has_leak, f"Memory leak detected: {severity} ({details['uss_growth_mb']:.1f} MB)"


@pytest.fixture
def memory_profiler():
    profiler = MemoryProfiler()
    profiler.start_profiling()
    yield profiler
    profiler.stop_profiling()
    gc.collect()


@pytest.fixture
def app_config():
    return GlobalAppConfig()


@pytest.fixture
async def event_bus():
    bus = EventBus()
    await bus.start_worker()
    yield bus
    await bus.stop_worker()


# =============================================================================
# COMPONENT-LEVEL MEMORY TESTS
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.memory
async def test_event_bus_memory_lifecycle(memory_profiler):
    async def test_eventbus(profiler):
        from iris.events.stt_events import CommandTextRecognizedEvent

        event_bus = EventBus()
        await event_bus.start_worker()
        await asyncio.sleep(0.1)

        # Publish events
        for i in range(50):
            event = CommandTextRecognizedEvent(text=f"cmd {i}", confidence=0.95, engine="test", processing_time_ms=10.0)
            await event_bus.publish(event)

        await event_bus.stop_worker()
        del event_bus
        gc.collect()
        await asyncio.sleep(0.1)

    await run_memory_test(memory_profiler, "EventBus", test_eventbus, max_growth_mb=10.0)


@pytest.mark.asyncio
@pytest.mark.memory
@pytest.mark.slow
async def test_whisper_stt_memory_lifecycle(memory_profiler, app_config):
    async def test_whisper(profiler):
        from iris.services.audio.whisper_stt import WhisperSpeechToText

        whisper = WhisperSpeechToText(model_name="base", device="cpu", sample_rate=16000, config=app_config)
        await asyncio.sleep(0.5)

        # Multiple recognitions
        dummy_audio = np.random.randint(-1000, 1000, size=16000*2, dtype=np.int16).tobytes()
        for _ in range(3):
            whisper.recognize(dummy_audio, 16000)
            gc.collect()

        if hasattr(whisper, 'shutdown'):
            await whisper.shutdown()
        del whisper
        gc.collect()
        await asyncio.sleep(0.5)

    await run_memory_test(memory_profiler, "WhisperSTT", test_whisper, max_growth_mb=150.0)


@pytest.mark.asyncio
@pytest.mark.memory
async def test_llm_service_memory_lifecycle(memory_profiler, event_bus, app_config):
    async def test_llm(profiler):
        from iris.services.audio.dictation_handling.llm_support.llm_service import LLMService
        from iris.services.storage.llm_model_downloader import LLMModelDownloader

        downloader = LLMModelDownloader(app_config)
        if not downloader.model_exists(app_config.llm.get_model_filename()):
            pytest.skip("LLM model not available")

        llm_service = LLMService(event_bus, app_config)
        if not await llm_service.initialize():
            pytest.skip("LLM init failed")

        await asyncio.sleep(1.0)

        # Process text
        test_texts = ["test text 1", "test text 2"]
        for text in test_texts:
            await llm_service.process_dictation(text, "Format this")
            await asyncio.sleep(0.5)
            gc.collect()

        await llm_service.shutdown()
        del llm_service
        gc.collect()
        await asyncio.sleep(1.0)

    await run_memory_test(memory_profiler, "LLMService", test_llm, max_growth_mb=500.0)


@pytest.mark.asyncio
@pytest.mark.memory
async def test_audio_buffer_memory_patterns(memory_profiler):
    async def test_audio(profiler):
        audio_chunks = []
        for i in range(50):
            chunk = np.random.randint(-1000, 1000, size=1600, dtype=np.int16).tobytes()
            audio_chunks.append(chunk)
            audio_float = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            _ = np.mean(audio_float)

            if len(audio_chunks) > 25:
                audio_chunks = audio_chunks[-25:]
            gc.collect()

        audio_chunks.clear()
        gc.collect()
        await asyncio.sleep(0.1)

    await run_memory_test(memory_profiler, "AudioBuffers", test_audio, max_growth_mb=5.0)


# =============================================================================
# INTEGRATION TESTS - FULL SYSTEM
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.memory
@pytest.mark.slow
async def test_service_initialization_memory(memory_profiler, app_config):
    async def test_services(profiler):
        from iris.services.storage.unified_storage_service import UnifiedStorageService
        from iris.services.storage.storage_adapters import StorageAdapterFactory
        from iris.services.grid.grid_service import GridService
        from iris.services.automation_service import AutomationService
        from iris.services.centralized_command_parser import CentralizedCommandParser

        event_bus = EventBus()
        await event_bus.start_worker()

        storage_service = UnifiedStorageService(event_bus, app_config)
        storage_factory = StorageAdapterFactory(storage_service)

        grid_service = GridService(event_bus, app_config)
        grid_service.setup_subscriptions()

        automation_service = AutomationService(event_bus, app_config)
        automation_service.setup_subscriptions()

        command_adapter = storage_factory.get_command_adapter()
        parser = CentralizedCommandParser(event_bus, app_config, command_adapter)
        await parser.initialize()
        parser.setup_subscriptions()

        await asyncio.sleep(0.5)

        # Cleanup
        services = [parser, automation_service, grid_service, storage_factory, storage_service, event_bus]
        for service in services:
            if hasattr(service, 'shutdown'):
                await service.shutdown()
        del services[:]
        gc.collect()
        await asyncio.sleep(0.5)

    await run_memory_test(memory_profiler, "ServiceInitialization", test_services, max_growth_mb=150.0)


