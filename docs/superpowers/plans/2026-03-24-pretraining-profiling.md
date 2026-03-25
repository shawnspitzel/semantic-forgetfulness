# Pretraining Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add always-on per-phase timing, GPU/CPU memory sampling, and throughput metrics to the pretraining loop, plus opt-in deep profiling via `--profile N` that produces a Chrome trace, a memory flamegraph, and a Plotly HTML dashboard.

**Architecture:** A new `TrainingProfiler` class in `src/utils/profiler.py` owns all profiling logic. It is composed of five internal helpers (`PhaseTimer`, `MemorySampler`, `ThroughputTracker`, `DiskWriter`, `ReportGenerator`) plus an optional `DeepProfiler` that wraps `torch.profiler`. `pretrain.py` instantiates it and calls `profiler.phase()`, `profiler.log_step()`, and `profiler.finish()`.

**Tech Stack:** Python 3.11, PyTorch >= 2.1, `psutil` (CPU RSS), `plotly` (HTML report), `torch.profiler` (deep mode), `torch.cuda._memory_viz` (flamegraph)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/utils/profiler.py` | All profiling logic — `PhaseTimer`, `MemorySampler`, `ThroughputTracker`, `DiskWriter`, `ReportGenerator`, `DeepProfiler`, `TrainingProfiler` |
| Modify | `pyproject.toml` | Add `profiling` optional dependency group |
| Modify | `src/training/pretrain.py` | Wire `TrainingProfiler` in, add `--profile N` arg |

---

## Task 1: Add profiling optional dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `profiling` optional dependency group**

In `pyproject.toml`, extend `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov"]
profiling = ["psutil>=5.9", "plotly>=5.0"]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add psutil and plotly as profiling optional deps"
```

---

## Task 2: PhaseTimer, MemorySampler, ThroughputTracker

**Files:**
- Create: `src/utils/profiler.py`

- [ ] **Step 1: Create `src/utils/profiler.py` with the three primitives**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/utils/profiler.py
git commit -m "feat: add PhaseTimer, MemorySampler, ThroughputTracker"
```

---

## Task 3: DiskWriter

**Files:**
- Modify: `src/utils/profiler.py`

- [ ] **Step 1: Append DiskWriter to `src/utils/profiler.py`**

```python
# ── DiskWriter ───────────────────────────────────────────────────────────────

class DiskWriter:
    """Appends newline-delimited JSON records to a file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    def write(self, record: dict) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
```

- [ ] **Step 2: Commit**

```bash
git add src/utils/profiler.py
git commit -m "feat: add DiskWriter"
```

---

## Task 4: ReportGenerator

**Files:**
- Modify: `src/utils/profiler.py`

- [ ] **Step 1: Append ReportGenerator to `src/utils/profiler.py`**

```python
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
            go.Scatter(x=steps, y=[r["throughput"]["tokens_per_sec"] for r in records],
                       name="tokens/sec", mode="lines"),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=steps, y=[r["throughput"]["segments_per_sec"] for r in records],
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
```

- [ ] **Step 2: Commit**

```bash
git add src/utils/profiler.py
git commit -m "feat: add ReportGenerator"
```

---

## Task 5: DeepProfiler

**Files:**
- Modify: `src/utils/profiler.py`

- [ ] **Step 1: Append DeepProfiler to `src/utils/profiler.py`**

```python
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
        except Exception:
            pass

    def _render_flamegraph(self, snapshot: object) -> None:
        try:
            from torch.cuda._memory_viz import trace_plot
            html = trace_plot(snapshot)
            (self._run_dir / "memory_flamegraph.html").write_text(html, encoding="utf-8")
        except Exception:
            pass
```

- [ ] **Step 2: Commit**

```bash
git add src/utils/profiler.py
git commit -m "feat: add DeepProfiler"
```

---

## Task 6: TrainingProfiler

**Files:**
- Modify: `src/utils/profiler.py`

- [ ] **Step 1: Append TrainingProfiler to `src/utils/profiler.py`**

```python
# ── TrainingProfiler ──────────────────────────────────────────────────────────

class TrainingProfiler:
    """
    Main entry point for training profiling.

    Usage:
        profiler = TrainingProfiler(use_wandb=True, profile_steps=0)
        with profiler.phase("data_load"):
            ...
        profiler.log_step(step, metrics_dict, n_tokens=len(seg), n_segments=1)
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
```

- [ ] **Step 2: Commit**

```bash
git add src/utils/profiler.py
git commit -m "feat: add TrainingProfiler"
```

---

## Task 7: Wire TrainingProfiler into pretrain.py

**Files:**
- Modify: `src/training/pretrain.py`

- [ ] **Step 1: Add import at top of `pretrain.py`**

After the existing imports, add:

```python
from utils.profiler import TrainingProfiler
```

- [ ] **Step 2: Update `train()` signature**

```python
def train(cfg: Config, data_path: Path, steps: int, device: str = "cpu",
          use_wandb: bool = False, profile_steps: int = 0) -> None:
```

- [ ] **Step 3: Add validation and profiler instantiation**

Immediately after `dev = torch.device(device)`:

```python
if profile_steps > 0 and steps < profile_steps + 3:
    raise ValueError(
        f"--profile {profile_steps} requires at least {profile_steps + 3} "
        f"total steps, but --steps {steps} was given."
    )

profiler = TrainingProfiler(use_wandb=use_wandb, profile_steps=profile_steps)
```

- [ ] **Step 4: Wrap phases in the training loop**

The existing `for` loop is nested inside an outer `while step < steps:` loop — **preserve both**. Replace only the inner `for` loop body (lines ~114–188). The phase wrappers are added around each logical block and `profiler.step()` (not `log_step`) is called unconditionally every step so the torch.profiler schedule advances correctly. `profiler.log_step()` replaces the old `wandb.log(...)` call:

```python
while step < steps:
    for seg_ids in _stream_segments(data_path, tokenizer, cfg):
        if step >= steps:
            break

    with profiler.phase("data_load"):
        seg_tensor = torch.tensor([seg_ids], device=dev)

    with profiler.phase("llm_forward_full"):
        with torch.no_grad():
            orig_embeds = embed(seg_tensor)[0]
            out_full = llm(seg_tensor, output_hidden_states=True)
            h_full_layers = {
                layer_idx: out_full.hidden_states[layer_idx][:, -1, :].detach()
                for layer_idx in layer_range
            }
            del out_full

    with profiler.phase("entity_extraction"):
        seg_text = tokenizer.decode(seg_ids)
        entities = extractor.extract(seg_text)
        sents = [s.strip() for s in seg_text.split(".") if s.strip()]
        anchors = SanityAnchors(
            boundary_sentences=[sents[0] if sents else "", sents[-1] if sents else ""],
            entities=entities,
            semantic_fingerprint=orig_embeds.mean(dim=0).detach().cpu(),
        )

    with profiler.phase("compressor"):
        ce_l2 = compressor.compress(orig_embeds, cfg.C_L2)
        ce_seq = ce_l2.unsqueeze(0)

    with profiler.phase("llm_forward_ce"):
        out_ce = llm(inputs_embeds=ce_seq, output_hidden_states=True)
        l_distill = torch.tensor(0.0, device=dev)
        layer_losses: dict[str, float] = {}
        for layer_idx in layer_range:
            h_full = h_full_layers[layer_idx]
            h_ce = out_ce.hidden_states[layer_idx][:, -1, :]
            std = h_full.std().clamp(min=1e-6)
            ll = F.smooth_l1_loss(h_ce / std, h_full / std)
            l_distill = l_distill + ll
            layer_losses[f"l_distill/layer_{layer_idx}"] = ll.item()
        l_distill = l_distill / max(len(layer_range), 1)
        del out_ce

    with profiler.phase("reconstructor"):
        result = reconstructor.reconstruct(ce_l2, anchors, [], None, "l3_to_l2")
        recon_mean = result.ce_tensor.mean(dim=0)
        orig_mean = orig_embeds.mean(dim=0)
        l_recon = 1.0 - F.cosine_similarity(
            recon_mean.unsqueeze(0), orig_mean.unsqueeze(0)
        ).mean()

    loss = l_distill + l_recon

    with profiler.phase("backward"):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

    with profiler.phase("optimizer"):
        optimizer.step()

    step += 1

    if step % 10 == 0:
        slot_norms = ce_l2.norm(dim=-1).detach()
        confidence_scores = (
            [s for _, s in result.confidence_scores]
            if result.confidence_scores else [0.0]
        )
        metrics = {
            "loss/total": loss.item(),
            "loss/l_distill": l_distill.item(),
            "loss/l_recon": l_recon.item(),
            "grad_norm/compressor": _grad_norm(compressor),
            "grad_norm/reconstructor": _grad_norm(reconstructor),
            "ce/slot_norm_mean": slot_norms.mean().item(),
            "ce/slot_norm_std": slot_norms.std().item(),
            "reconstruction/fingerprint_sim": (
                result.fingerprint_sim if result.fingerprint_sim is not None else 0.0
            ),
            "reconstruction/confidence_mean": sum(confidence_scores) / len(confidence_scores),
            "reconstruction/confidence_min": min(confidence_scores),
            "reconstruction/fallback": float(result.fallback),
            "reconstruction/grounding_used": float(result.grounding_used),
            **layer_losses,
        }
        profiler.log_step(step, metrics, n_tokens=len(seg_ids), n_segments=1)

        if step % 50 == 0:
            print(f"Step {step}/{steps}  L_distill={l_distill.item():.4f}  L_recon={l_recon.item():.4f}")

        profiler.step()   # must be outside any cadence check — advances torch.profiler every step
```

Notes:
- The old `if step % 10 == 0 and use_wandb: wandb.log(...)` block is **removed** — replaced by `profiler.log_step(...)` (which handles W&B internally).
- The old `if use_wandb: wandb.finish()` block at the end of `train()` is **removed** — `profiler.finish()` calls `wandb.finish()` internally.
- `profiler.step()` must be called every training step (not just at log cadence) so the torch.profiler schedule counts steps correctly.

- [ ] **Step 5: Call `profiler.finish()` after checkpoint save**

```python
Path("checkpoints").mkdir(exist_ok=True)
compressor.model.save_pretrained("checkpoints/compressor")
reconstructor.model.save_pretrained("checkpoints/reconstructor")
print("Adapters saved to checkpoints/")

profiler.finish()
print(f"Profiling artifacts: {profiler.run_dir}/")
```

- [ ] **Step 6: Add `--profile` argument in `__main__` block**

After the existing `parser.add_argument("--wandb", ...)` line:

```python
parser.add_argument("--profile", type=int, default=0,
                    metavar="N", help="Profile N steps with torch.profiler (0 = disabled)")
```

Update the `train(...)` call:

```python
train(Config.load(), args.data_path, args.steps, args.device,
      use_wandb=args.wandb, profile_steps=args.profile)
```

- [ ] **Step 7: Commit**

```bash
git add src/training/pretrain.py
git commit -m "feat: wire TrainingProfiler into pretrain.py"
```

---

## Verification

```bash
# Smoke test: 15 steps with deep profiling enabled
python -m sf.training.pretrain \
  --data-path data/train.txt \
  --steps 15 \
  --device cpu \
  --profile 3
```

Check artifacts:

```bash
ls observability/run_*/
# Expected: metrics.jsonl  report.html  trace.json
# (memory_flamegraph.html and memory_snapshot.pickle only on CUDA)
```

Open `observability/run_<timestamp>/report.html` in a browser to verify the four-panel Plotly dashboard renders correctly.
