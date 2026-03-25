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


# ── ReportGenerator ───────────────────────────────────────────────────────────

class ReportGenerator:
    """Reads metrics.jsonl and generates a self-contained Plotly HTML report."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir

    def generate(self) -> None:
        records = self._load_records()
        if not records:
            return
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except Exception:
            return

        steps = [r["step"] for r in records]
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                "Memory (MB)", "Per-Phase Timing (ms, stacked)",
                "Throughput", "Loss & Grad Norms",
            ],
            vertical_spacing=0.08,
        )

        # Panel 1 — memory
        for key, label in [
            ("gpu_allocated_mb", "GPU Allocated"),
            ("gpu_reserved_mb", "GPU Reserved"),
            ("gpu_peak_mb", "GPU Peak"),
            ("cpu_rss_mb", "CPU RSS"),
        ]:
            vals = [r["memory"].get(key) for r in records]
            if any(v is not None for v in vals):
                fig.add_trace(go.Scatter(x=steps, y=vals, name=label, mode="lines"), row=1, col=1)

        # Panel 2 — per-phase stacked bar
        phases = list(records[0].get("phase_ms", {}).keys())
        for phase in phases:
            fig.add_trace(
                go.Bar(x=steps, y=[r["phase_ms"].get(phase, 0.0) for r in records], name=phase),
                row=2, col=1,
            )

        # Panel 3 — throughput
        fig.add_trace(
            go.Scatter(x=steps, y=[r.get("throughput", {}).get("tokens_per_sec") for r in records],
                       name="tokens/sec", mode="lines"),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=steps, y=[r.get("throughput", {}).get("segments_per_sec") for r in records],
                       name="segments/sec", mode="lines"),
            row=3, col=1,
        )

        # Panel 4 — losses + grad norms
        all_keys: set[str] = set()
        for r in records:
            all_keys.update(r.get("metrics", {}).keys())
        for key in sorted(all_keys):
            fig.add_trace(
                go.Scatter(x=steps, y=[r.get("metrics", {}).get(key) for r in records],
                           name=key, mode="lines"),
                row=4, col=1,
            )

        fig.update_layout(barmode="stack", title="Training Profiling Report", height=1600)
        fig.write_html(str(self._run_dir / "report.html"))

    def _load_records(self) -> list[dict]:
        path = self._run_dir / "metrics.jsonl"
        if not path.exists():
            return []
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records


# ── DeepProfiler ──────────────────────────────────────────────────────────────

class DeepProfiler:
    """
    Wraps torch.profiler for deep opt-in profiling.

    Captures N active steps using schedule(wait=1, warmup=2, active=N),
    then exports a Chrome trace and renders a memory flamegraph (CUDA only).
    Only instantiated when profile_steps > 0.
    """

    def __init__(self, run_dir: Path, profile_steps: int) -> None:
        from torch.profiler import ProfilerActivity, profile, schedule

        self._run_dir = run_dir
        self._profile_steps = profile_steps
        self._total_steps_needed = 1 + 2 + profile_steps
        self._steps_taken = 0
        self._snapshot_taken = False
        self._active = False

        self._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=2, active=profile_steps),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        )

    def start(self) -> None:
        self._profiler.__enter__()
        self._active = True

    def step(self) -> None:
        if not self._active:
            return
        self._profiler.step()
        self._steps_taken += 1
        if self._steps_taken >= self._total_steps_needed and not self._snapshot_taken:
            self._capture_memory_snapshot()

    def finish(self) -> None:
        if not self._active:
            return
        self._profiler.__exit__(None, None, None)
        self._active = False
        self._profiler.export_chrome_trace(str(self._run_dir / "trace.json"))
        if not self._snapshot_taken:
            self._capture_memory_snapshot()

    def _capture_memory_snapshot(self) -> None:
        self._snapshot_taken = True
        if not torch.cuda.is_available():
            return
        try:
            import pickle
            snapshot = torch.cuda.memory._snapshot()
            snap_path = self._run_dir / "memory_snapshot.pickle"
            with snap_path.open("wb") as f:
                pickle.dump(snapshot, f)
            self._render_flamegraph(snapshot)
        except Exception as e:
            print(f"[profiler] memory snapshot skipped: {e}")

    def _render_flamegraph(self, snapshot: object) -> None:
        try:
            from torch.cuda._memory_viz import trace_plot
            html = trace_plot(snapshot)
            (self._run_dir / "memory_flamegraph.html").write_text(html, encoding="utf-8")
        except Exception:
            pass


# ── TrainingProfiler ──────────────────────────────────────────────────────────

class TrainingProfiler:
    """
    Main entry point for training profiling.

    Usage:
        profiler = TrainingProfiler(use_wandb=True, profile_steps=0)
        with profiler.phase("data_load"):
            ...
        profiler.log_step(step, metrics_dict, n_tokens=len(seg), n_segments=1)
        profiler.step()   # call every training step
        profiler.finish()   # call after checkpoint save
    """

    def __init__(self, use_wandb: bool, profile_steps: int = 0) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path("observability") / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._use_wandb = use_wandb
        self._log_cadence = 10
        self._phase_timer = PhaseTimer()
        self._memory_sampler = MemorySampler()
        self._throughput_tracker = ThroughputTracker()
        self._disk_writer = DiskWriter(self.run_dir / "metrics.jsonl")
        self._report_generator = ReportGenerator(self.run_dir)
        self._deep_profiler: DeepProfiler | None = (
            DeepProfiler(self.run_dir, profile_steps) if profile_steps > 0 else None
        )

        if self._deep_profiler is not None:
            self._deep_profiler.start()

    def phase(self, name: str) -> contextlib.AbstractContextManager[None]:
        return self._phase_timer.phase(name)

    def log_step(self, step: int, metrics: dict, n_tokens: int, n_segments: int) -> None:
        self._throughput_tracker.record(n_tokens, n_segments)

        if step % self._log_cadence != 0:
            return

        phase_ms = self._phase_timer.flush()
        memory = self._memory_sampler.sample()
        throughput = self._throughput_tracker.flush()

        record = {
            "step": step,
            "wall_time": time.time(),
            "phase_ms": phase_ms,
            "memory": memory,
            "throughput": throughput,
            "metrics": metrics,
        }
        self._disk_writer.write(record)

        if self._use_wandb:
            try:
                import wandb
                flat: dict = {
                    **{f"timing/{k}": v for k, v in phase_ms.items()},
                    **{f"memory/{k}": v for k, v in memory.items() if v is not None},
                    **{f"throughput/{k}": v for k, v in throughput.items()},
                    **metrics,
                }
                wandb.log(flat, step=step)
            except ImportError:
                pass

    def step(self) -> None:
        """Call once per training step (every step, not just log steps) for deep profiler."""
        if self._deep_profiler is not None:
            self._deep_profiler.step()

    def finish(self) -> None:
        if self._deep_profiler is not None:
            self._deep_profiler.finish()
        self._report_generator.generate()
        if self._use_wandb:
            try:
                import wandb
                wandb.finish()
            except ImportError:
                pass
