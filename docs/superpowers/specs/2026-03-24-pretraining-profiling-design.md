# Pretraining Loop — In-Depth Memory & Time Profiling

**Date:** 2026-03-24
**Status:** Approved

---

## 1. Problem Statement

The current `pretrain.py` training loop has no memory or timing visibility. The only
observability is a console print every 50 steps and W&B loss/grad-norm metrics. This
makes it impossible to:

- Identify which phases dominate step time
- Detect GPU memory pressure before OOM crashes
- Understand peak allocation patterns across the training loop
- Diagnose throughput regressions after code changes

---

## 2. Goals

1. Always-on lightweight profiling: per-phase wall-clock timing, GPU/CPU memory
   sampling, and throughput metrics written to disk at the W&B log cadence (every 10 steps)
2. Deep opt-in profiling via `--profile N`: full `torch.profiler` trace + memory
   snapshot for N steps, producing a Chrome trace and a PyTorch memory flamegraph
3. Post-run HTML report: interactive time-series charts auto-generated from the
   always-on metrics, viewable offline without W&B
4. Reusable across `pretrain.py` and `finetune.py`

---

## 3. Architecture

### 3.1 New file: `src/utils/profiler.py`

Contains `TrainingProfiler` — the single entry point for all profiling logic.

**Public interface:**

```python
class TrainingProfiler:
    def __init__(self, run_dir: Path, use_wandb: bool, profile_steps: int = 0): ...
    def phase(self, name: str) -> ContextManager: ...
    def log_step(self, step: int, metrics: dict, n_tokens: int, n_segments: int): ...
    def finish(self): ...
```

- `phase(name)` — context manager that times the named block with `time.perf_counter`
- `log_step(...)` — called at the W&B log cadence; handles disk write, W&B push, memory sample, throughput calculation, and `torch.profiler` step
- `finish()` — flushes remaining records, generates `report.html`, generates `memory_flamegraph.html` (if `--profile`), exports Chrome trace (if `--profile`)

**Internal components:**

| Component | Responsibility |
|---|---|
| `PhaseTimer` | Accumulates wall-clock time per named phase between log steps |
| `MemorySampler` | Reads `torch.cuda.memory_allocated/reserved/max_memory_allocated` + `psutil` CPU RSS |
| `ThroughputTracker` | Counts tokens and segments processed; computes rate at each log step |
| `DiskWriter` | Appends one JSONL record per log step to `metrics.jsonl` |
| `ReportGenerator` | Reads `metrics.jsonl` at run end, generates `report.html` via Plotly |
| `DeepProfiler` | Wraps `torch.profiler.profile`; activated only when `profile_steps > 0` |

### 3.2 Changes to `pretrain.py`

- Add `--profile N` CLI argument (default `0` = disabled)
- Instantiate `TrainingProfiler(run_dir, use_wandb, profile_steps)` before the loop
- Wrap each logical phase:
  ```python
  with profiler.phase("data_load"): ...
  with profiler.phase("llm_forward_full"): ...
  with profiler.phase("llm_forward_ce"): ...
  with profiler.phase("compressor"): ...
  with profiler.phase("reconstructor"): ...
  with profiler.phase("backward"): ...
  with profiler.phase("optimizer"): ...
  ```
- Replace bare `wandb.log(...)` with `profiler.log_step(step, metrics, n_tokens, n_segments)`
- Call `profiler.finish()` before checkpoint save

---

## 4. Disk Layout

```
observability/
└── run_20260324_143022/          # ISO timestamp at train() start
    ├── metrics.jsonl             # always-on: one record per log step
    ├── report.html               # always-on: auto-generated Plotly dashboard
    ├── trace.json                # --profile only: Chrome trace
    └── memory_flamegraph.html   # --profile only: PyTorch memory flamegraph
```

### 4.1 `metrics.jsonl` record schema

One newline-delimited JSON object per log step:

```json
{
  "step": 10,
  "wall_time": 1742823045.3,
  "phase_ms": {
    "data_load": 12.4,
    "llm_forward_full": 340.1,
    "llm_forward_ce": 280.5,
    "compressor": 18.2,
    "reconstructor": 9.7,
    "backward": 155.3,
    "optimizer": 4.1
  },
  "memory": {
    "gpu_allocated_mb": 4821.3,
    "gpu_reserved_mb": 5120.0,
    "gpu_peak_mb": 5204.7,
    "cpu_rss_mb": 2341.0
  },
  "throughput": {
    "tokens_per_sec": 1842.3,
    "segments_per_sec": 92.1
  },
  "metrics": {
    "loss/total": 0.412,
    "loss/l_distill": 0.301,
    "loss/l_recon": 0.111,
    "grad_norm/compressor": 1.23
  }
}
```

`gpu_peak_mb` reflects the worst single step in the log window. `torch.cuda.reset_peak_memory_stats()` is called after each log step so the peak resets per window.

---

## 5. Report Artifacts

### 5.1 `report.html` — time-series dashboard

Generated from `metrics.jsonl` by `ReportGenerator` using Plotly. Four panels:

1. **Memory over steps** — `gpu_allocated_mb`, `gpu_reserved_mb`, `gpu_peak_mb`, `cpu_rss_mb` as overlapping line chart
2. **Per-phase timing** — stacked bar chart, one bar per log step, each segment is a named phase
3. **Throughput** — `tokens_per_sec` and `segments_per_sec` as dual-axis line chart
4. **Loss + grad norms** — mirrors W&B dashboard for offline viewing

Self-contained HTML with Plotly loaded from CDN. Always generated, even without `--profile`.

### 5.2 `memory_flamegraph.html` — allocation flamegraph

Generated from the memory snapshot pickle via `torch.cuda._memory_viz.snapshot_plot()`.
Shows tensor-level GPU allocations on a timeline — which operation allocated what,
and whether it was freed. Answers "what is holding memory" rather than "how much memory."

Only generated when `--profile N` is passed.

### 5.3 `trace.json` — Chrome trace

Exported via `torch.profiler`'s built-in Chrome trace exporter. Load in
`chrome://tracing` or Perfetto UI for kernel-level flame graph of CPU + CUDA ops.

Only generated when `--profile N` is passed.

---

## 6. Deep Profile Mode

Activated by `--profile N`. Profiles exactly N steps using:

```python
torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=N),
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
)
```

The `wait=1, warmup=2` schedule discards the first 3 steps (startup noise) before
capturing N active steps. `profile_memory=True` enables per-tensor allocation tracking.
`with_stack=True` enables Python stack frames in the trace — required for the
flamegraph to be readable.

A `torch.cuda.memory._snapshot()` is taken at the end of the profile window and
saved as `memory_snapshot.pickle`, then rendered to `memory_flamegraph.html`.

---

## 7. Throughput Calculation

`ThroughputTracker` accumulates `n_tokens` and `n_segments` between log steps.
At each `log_step` call:

```
tokens_per_sec  = total_tokens_since_last_log / elapsed_wall_seconds
segments_per_sec = total_segments_since_last_log / elapsed_wall_seconds
```

Both counters and the wall clock reset after each log. This gives per-window
throughput, not a global average — useful for detecting slowdowns mid-run.

---

## 8. Dependencies

| Package | Already present | Notes |
|---|---|---|
| `torch` | yes | `torch.profiler`, `torch.cuda.memory_*` |
| `psutil` | needs adding | CPU RSS tracking |
| `plotly` | needs adding | `report.html` generation |
| `wandb` | yes (optional) | unchanged |

Both `psutil` and `plotly` should be added to `setup.cfg` / `requirements` as
optional dev dependencies (`pip install ".[profiling]"` or similar).

---

## 9. Out of Scope

- Inference-time profiling (`inference_loop.py`, `cache_controller.py`) — separate feature
- `finetune.py` integration — `TrainingProfiler` is designed to support this but the
  wiring is deferred; `finetune.py` is not touched in this implementation
- W&B custom charts — the existing W&B log call is extended with memory/timing
  metrics; no W&B dashboard configuration changes are required
