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
   sampling, and throughput metrics written to disk every 10 steps (independent of W&B)
2. Deep opt-in profiling via `--profile N`: full `torch.profiler` trace capturing N
   active steps + memory snapshot, producing a Chrome trace and a PyTorch memory
   flamegraph
3. Post-run HTML report: interactive time-series charts auto-generated from the
   always-on metrics, viewable offline without W&B
4. Reusable across `pretrain.py` and `finetune.py`

**Minimum PyTorch version required: 2.1** (for `torch.cuda.memory._snapshot()` and
the associated memory visualization tooling).

---

## 3. Architecture

### 3.1 New file: `src/utils/profiler.py`

Contains `TrainingProfiler` — the single entry point for all profiling logic.

**Public interface:**

```python
class TrainingProfiler:
    def __init__(self, use_wandb: bool, profile_steps: int = 0): ...
    def phase(self, name: str) -> contextlib.AbstractContextManager[None]: ...
    def log_step(self, step: int, metrics: dict, n_tokens: int, n_segments: int): ...
    def finish(self): ...
```

- `TrainingProfiler.__init__` creates `observability/run_<ISO_timestamp>/` internally.
  The caller does not provide a path. The run directory is available as `profiler.run_dir`
  after construction.
- `phase(name)` — context manager that times the named block with `time.perf_counter`.
  Phases accumulate between `log_step` calls and are reset after each flush.
- `log_step(...)` — called every 10 steps regardless of whether W&B is enabled. Handles:
  disk write, W&B push (if `use_wandb=True`), memory sample, throughput calculation,
  and `torch.profiler` step (if deep profiling active). W&B logging is gated
  independently inside this method; disk writes always occur.
- `finish()` — called after checkpoint save. Flushes remaining records, generates
  `report.html`, and (if `--profile`) exports the Chrome trace and renders
  `memory_flamegraph.html`.

**Internal components:**

| Component | Responsibility |
|---|---|
| `PhaseTimer` | Accumulates wall-clock time per named phase between log steps |
| `MemorySampler` | Reads GPU memory stats (guarded by `torch.cuda.is_available()`) + `psutil` CPU RSS |
| `ThroughputTracker` | Counts tokens and segments processed; computes rate at each log step |
| `DiskWriter` | Appends one JSONL record per log step to `metrics.jsonl` |
| `ReportGenerator` | Reads `metrics.jsonl` at run end, generates `report.html` via Plotly |
| `DeepProfiler` | Wraps `torch.profiler.profile`; instantiated **only** when `profile_steps > 0` |

### 3.2 Changes to `pretrain.py`

- Add `--profile N` CLI argument (default `0` = disabled)
- Add validation: if `profile_steps > 0`, assert `steps >= profile_steps + 3`
  (the `torch.profiler` schedule requires `wait=1, warmup=2, active=N` = N+3 total
  steps to complete one capture cycle; fail fast with a clear message if the run is
  too short)
- Instantiate `TrainingProfiler(use_wandb, profile_steps)` before the loop
- Wrap phases in execution order:
  ```python
  with profiler.phase("data_load"): ...           # _stream_segments yield
  with profiler.phase("llm_forward_full"): ...    # frozen LLM on orig tokens
  with profiler.phase("entity_extraction"): ...   # extractor.extract + anchors build
  with profiler.phase("compressor"): ...          # compressor.compress
  with profiler.phase("llm_forward_ce"): ...      # frozen LLM on CE embeddings
  with profiler.phase("reconstructor"): ...       # reconstructor.reconstruct
  with profiler.phase("backward"): ...            # loss.backward + clip_grad_norm_
  with profiler.phase("optimizer"): ...           # optimizer.step + zero_grad
  ```
- Replace bare `wandb.log(...)` with `profiler.log_step(step, metrics, n_tokens, n_segments)`
- Call `profiler.finish()` **after** checkpoint save (so checkpoint I/O is included
  in the final timing record)

---

## 4. Disk Layout

```
observability/
└── run_20260324_143022/          # ISO timestamp at TrainingProfiler.__init__
    ├── metrics.jsonl             # always-on: one record per log step
    ├── report.html               # always-on: auto-generated Plotly dashboard
    ├── trace.json                # --profile only: Chrome trace
    ├── memory_snapshot.pickle   # --profile only: raw snapshot for offline analysis
    └── memory_flamegraph.html   # --profile only: PyTorch memory flamegraph
```

The `observability/` root and the run subdirectory are created by `TrainingProfiler.__init__`
via `Path.mkdir(parents=True, exist_ok=True)`.

### 4.1 `metrics.jsonl` record schema

One newline-delimited JSON object per log step:

```json
{
  "step": 10,
  "wall_time": 1742823045.3,
  "phase_ms": {
    "data_load": 12.4,
    "llm_forward_full": 340.1,
    "entity_extraction": 8.3,
    "compressor": 18.2,
    "llm_forward_ce": 280.5,
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

**GPU memory fields:** `MemorySampler` checks `torch.cuda.is_available()` before
calling any `torch.cuda.*` function. When running CPU-only, all `gpu_*` fields are
`null` in the JSON record.

**`gpu_peak_mb`:** Reflects the worst single step in the current log window.
`torch.cuda.reset_peak_memory_stats()` is called after each log step so the peak
resets per window. Note: the first window (steps 1–10) includes model loading and
first-forward overhead and may report a peak significantly higher than steady-state.
This is expected and documented in the report.

---

## 5. Report Artifacts

### 5.1 `report.html` — time-series dashboard (always generated)

Built from `metrics.jsonl` by `ReportGenerator` using Plotly. Four panels:

1. **Memory over steps** — `gpu_allocated_mb`, `gpu_reserved_mb`, `gpu_peak_mb`,
   `cpu_rss_mb` as overlapping line chart (GPU fields hidden/noted as N/A on CPU runs)
2. **Per-phase timing** — stacked bar chart, one bar per log step, each segment is a
   named phase in execution order
3. **Throughput** — `tokens_per_sec` and `segments_per_sec` as dual-axis line chart
4. **Loss + grad norms** — mirrors W&B dashboard for offline viewing

Self-contained HTML with Plotly loaded from CDN. Always generated regardless of
`--profile`.

### 5.2 `memory_flamegraph.html` — allocation flamegraph (`--profile` only)

Generated from `memory_snapshot.pickle` using PyTorch's documented memory visualization
tooling (`torch.cuda.memory._snapshot()` to capture, then the official
`torch/cuda/_memory_viz.py` renderer). Shows tensor-level GPU allocations on a
timeline — which operation allocated what and whether it was freed. Answers "what is
holding memory" rather than "how much memory."

Only generated when `--profile N` is passed.

### 5.3 `trace.json` — Chrome trace (`--profile` only)

Exported via `torch.profiler`'s built-in Chrome trace exporter. Load in
`chrome://tracing` or Perfetto UI for kernel-level flamegraph of CPU + CUDA ops.

Only generated when `--profile N` is passed.

---

## 6. Deep Profile Mode

Activated by `--profile N`. `DeepProfiler` is **not instantiated** when
`profile_steps == 0` — no `torch.profiler.profile` context is entered and no overhead
is incurred.

When active, profiles using:

```python
torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=N),
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
)
```

**Schedule semantics:** The profiler requires `1 + 2 + N = N+3` total `.step()` calls
to complete one capture cycle. The `wait=1` phase discards step 1, `warmup=2` runs
steps 2–3 (no data collected), and `active=N` captures steps 4 through N+3. This
is why `pretrain.py` must validate `steps >= profile_steps + 3` at startup.

`torch.cuda.memory._snapshot()` is called at the end of the active window and saved
as `memory_snapshot.pickle`. The flamegraph renderer is invoked from `profiler.finish()`.

---

## 7. Throughput Calculation

`ThroughputTracker` accumulates `n_tokens` and `n_segments` between log steps. At
each `log_step` call:

```
tokens_per_sec   = total_tokens_since_last_log / elapsed_wall_seconds
segments_per_sec = total_segments_since_last_log / elapsed_wall_seconds
```

Both counters and the wall clock reset after each log. This gives per-window
throughput rather than a global average — useful for detecting slowdowns mid-run.

---

## 8. Dependencies

| Package | Already present | Notes |
|---|---|---|
| `torch >= 2.1` | yes | `torch.profiler`, `torch.cuda.memory._snapshot()` |
| `psutil` | needs adding | CPU RSS tracking |
| `plotly` | needs adding | `report.html` generation |
| `wandb` | yes (optional) | unchanged |

Both `psutil` and `plotly` should be added to `setup.cfg` / `pyproject.toml` as
optional dev dependencies under a `profiling` extra:
```
pip install ".[profiling]"
```

---

## 9. Out of Scope

- Inference-time profiling (`inference_loop.py`, `cache_controller.py`) — separate feature
- `finetune.py` integration — `TrainingProfiler` is designed to support this (same API)
  but the wiring is deferred; `finetune.py` is not touched in this implementation
- W&B dashboard configuration changes — existing W&B log calls are extended with
  memory/timing metrics; no W&B project configuration is required
