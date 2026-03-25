"""
Training profiler — memory, timing, and throughput instrumentation.
"""
from __future__ import annotations

import contextlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

import torch


# ── PhaseTimer ───────────────────────────────────────────────────────────────

class PhaseTimer:
    """Accumulates wall-clock time (ms) per named phase between flushes."""

    def __init__(self) -> None:
        self._acc: dict[str, float] = {}

    @contextlib.contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._acc[name] = self._acc.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0

    def flush(self) -> dict[str, float]:
        result = dict(self._acc)
        self._acc.clear()
        return result


# ── MemorySampler ─────────────────────────────────────────────────────────────

class MemorySampler:
    """Samples GPU and CPU memory. GPU fields are None when CUDA is unavailable."""

    def sample(self) -> dict[str, float | None]:
        result: dict[str, float | None] = {
            "gpu_allocated_mb": None,
            "gpu_reserved_mb": None,
            "gpu_peak_mb": None,
            "cpu_rss_mb": None,
        }
        try:
            import psutil
            result["cpu_rss_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        if torch.cuda.is_available():
            result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            result["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            result["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        return result


# ── ThroughputTracker ─────────────────────────────────────────────────────────

class ThroughputTracker:
    """Tracks tokens and segments processed; computes per-window rates on flush."""

    def __init__(self) -> None:
        self._tokens = 0
        self._segments = 0
        self._t0 = time.perf_counter()

    def record(self, n_tokens: int, n_segments: int) -> None:
        self._tokens += n_tokens
        self._segments += n_segments

    def flush(self) -> dict[str, float]:
        elapsed = max(time.perf_counter() - self._t0, 1e-9)
        result = {
            "tokens_per_sec": self._tokens / elapsed,
            "segments_per_sec": self._segments / elapsed,
        }
        self._tokens = 0
        self._segments = 0
        self._t0 = time.perf_counter()
        return result


# ── DiskWriter ───────────────────────────────────────────────────────────────

class DiskWriter:
    """Appends newline-delimited JSON records to a file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    def write(self, record: dict) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
