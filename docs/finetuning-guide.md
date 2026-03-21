# Semantic Forgetfulness — Fine-Tuning Guide

A complete guide to Phase 2 (end-of-session online learning) and the path toward
Phase 3 (mid-inference updates). Covers the miss event lifecycle, current vs. full
implementation, per-session monitoring, multi-session dynamics, and hyperparameter
guidance.

For Phase 1 (offline pretraining), see [pretraining-guide.md](pretraining-guide.md).
For the quick operational walkthrough, see [training-guide.md](training-guide.md).

---

## 1. What fine-tuning means in this system

Fine-tuning here is fundamentally different from conventional supervised fine-tuning.
There is no labeled dataset. The training signal comes entirely from the system's own
deployment failures — segments that were compressed into CEs that turned out to be
semantically insufficient for retrieval.

The core insight: **a cache miss is a retrospective label**. When a segment that was
needed is not found in L2 or L3 (or is found but fails the fingerprint gate), the
system has evidence that the CE produced for that segment at compression time was
inadequate. Re-running distillation on those specific segments teaches the compressor
to produce better CEs for them — not by penalizing over-compression globally, but by
providing a targeted gradient signal on exactly the segments that caused failures.

This is the only system that converts deployment failures into first-class training
examples while the model is deployed.

---

## 2. Miss event lifecycle

Understanding the full lifecycle from inference miss to gradient update is essential
for debugging and improving the fine-tuning signal.

### 2.1 MissEvent (lightweight, collected during session)

Logged in real time by `cache_controller.py` whenever a retrieval fails:

```python
@dataclass
class MissEvent:
    segment_id: uuid.UUID
    miss_type: MissType          # "soft" (L2) | "hard" (L3) | "total" (not found)
    query_vec: torch.Tensor      # [D] MiniLM fingerprint of query that triggered miss
    timestamp: float
```

These are accumulated in `cache_controller.miss_log` throughout the session.
They are lightweight by design — only the query fingerprint, segment ID, and timing.
The expensive tensor data is not stored here.

### 2.2 FullMissEvent (reconciled at session end)

At session end (triggered by `/done` or process exit), the cache controller reconciles
each `MissEvent` into a `FullMissEvent` by looking up the stored tensors for each
missed segment:

```python
@dataclass
class FullMissEvent:
    segment_input: torch.Tensor     # L2 miss: [N, D] original token embeddings
                                    # L3 miss: [C_L2, D] L2 CE at demotion time
    context_window: torch.Tensor    # [W, D] surrounding tokens at encoding time
    ce_produced: torch.Tensor       # [C_L2, D] or [C_L3, D] the CE that failed
    miss_level: "l2" | "l3"
    session_position: int
```

The distinction between L2 and L3 miss inputs is critical:
- **L2 miss**: The original token embeddings `[N, D]` are still available (the segment
  was in L2 as a CE, but the *original tokens* are stored separately in L3's
  `l2_ce_at_demotion`). The full distillation objective applies.
- **L3 miss**: Only the L2 CE tensor `[C_L2, D]` is available — the original tokens
  are gone. L_distill cannot be applied (no token sequence to run the teacher on).
  Only L_recon and L_inject apply.

### 2.3 Reconciliation gap

Not every `MissEvent` can be reconciled into a `FullMissEvent`. A `MissEvent` has
only a `segment_id`. If that segment was evicted from L3 before session end (due to
L3 capacity limits), the tensor data is lost and the miss event is silently dropped.

**Implication:** Sessions with heavy L3 eviction (very long sessions near `l3_capacity`
limit) produce fewer fine-tuning examples than their miss count would suggest. If you
see `[Fine-tuning: N events]` significantly lower than the miss count reported by
`/stats`, eviction is the cause. Consider increasing `l3_capacity` for training runs.

---

## 3. Current implementation vs. full specification

There is a deliberate gap between the architecture specification
(`docs/architecture/pretraining.md`) and the current `finetune.py`. Understanding
this gap is important for knowing what signal Phase 2 is actually providing today
and what it will provide when fully implemented.

### 3.1 What `finetune.py` currently does (MVP)

```python
for event in miss_events:
    inp = event.segment_input.to(device)
    target = inp.mean(dim=0)                    # mean of input (token embeds or L2 CE)
    target_c = cfg.C_L2 if event.miss_level == "l2" else cfg.C_L3
    ce = compressor.compress(inp, target_c)
    loss = 1.0 - cosine_similarity(ce.mean(dim=0), target.mean(dim=0))
```

This is a **simplified L_recon only**: it measures cosine distance between the
mean of the newly produced CE and the mean of the original input. Both L2 and L3
misses use the same loss function.

**What is missing from the current implementation:**

| Component | Missing in MVP | Impact |
|---|---|---|
| Teacher forward pass (L_distill for L2 misses) | `context_window` stored but not used | L2 miss gradient is weaker — no hidden-state alignment |
| Reconstructor in the loss | Compressor output not passed through reconstructor | L_recon doesn't validate the full compress → reconstruct roundtrip |
| L_inject | No task supervision | No verification that CE injection is task-coherent |
| Per-tier loss separation | Same loss for L2 and L3 misses | L3 misses cannot get L_distill by design, but L2 misses should |

### 3.2 What the full specification calls for

**For L2 misses** (original token embeddings available):

```
1. Re-run frozen LLM on (context_window + segment_input) → H_full at mid-to-late layers
2. Compress segment_input → CE_new  (LoRA forward pass)
3. Run frozen LLM with CE_new injected → H_compressed at same layers
4. L_distill = SmoothL1(H_compressed / std, H_full / std)
5. Reconstruct CE_new → recon_embeds
6. L_recon = 1 - cosine_similarity(recon_embeds.mean(), segment_input.mean())
7. (Optional) L_inject: run LLM on query with CE context, compare to teacher response
8. Loss = L_distill + L_recon [+ L_inject]
```

**For L3 misses** (only L2 CE available, no original tokens):

```
1. Compress L2_CE → CE_new at C_L3 slots
2. Reconstruct CE_new → recon_embeds
3. L_recon = 1 - cosine_similarity(recon_embeds.mean(), L2_CE.mean())
   Note: this targets L2 CE fidelity, not original token fidelity
4. (Optional) L_inject: same as above
5. Loss = L_recon [+ L_inject]
```

The `context_window` stored in `FullMissEvent` is the key to the L_distill upgrade.
It represents the W=100 token window surrounding the segment at the time of original
compression — sufficient to provide meaningful teacher context for a per-segment
distillation step.

### 3.3 Upgrading finetune.py to the full spec

When you are ready to implement the full fine-tuning objective, the entry point is
`run_finetune()` in `src/training/finetune.py`. The upgrade requires:

1. Pass the frozen `llm` and `layer_range` into `run_finetune()`
2. For each L2 miss event, run the teacher pass using `event.context_window`
3. Replace the simplified cosine loss with L_distill + proper L_recon (through reconstructor)

The reconstructor is already imported but not called in the current implementation.
The `Reconstructor` instance is passed in — it just needs to be used.

**Note:** The teacher forward pass in Phase 2 uses only `context_window_W=100` tokens
of context, not the full document like Phase 1. This makes Phase 2 teacher targets
weaker than Phase 1. This asymmetry is accepted — Phase 2 is refinement, not
re-pretraining.

---

## 4. Fine-tuning hyperparameters

### 4.1 `min_session_length` (default: 20)

The minimum number of reconciled `FullMissEvent` objects required before the gradient
step fires. Too low = gradient update from noise (1-2 noisy examples can overfit
immediately). Too high = sessions that would have provided useful signal are skipped.

**Calibration guidance:**

| Session type | Expected miss events | Recommended minimum |
|---|---|---|
| Short conversation (< 8K tokens) | 0 — L2/L3 never activates | N/A (L2 activates at 8K tokens) |
| Medium conversation (8K–32K tokens) | 5–30 depending on topic breadth | Start at 15 |
| Long conversation (32K–128K tokens) | 30–200 | 20–30 |
| Very long / agentic (128K+) | 200+ | 50 (consider mid-inference updates) |

The 8K and 32K thresholds are set by `l2_activation_threshold` and `l3_activation_threshold`
in `config.yaml`. Sessions shorter than 8K produce zero miss events by design — no
fine-tuning fires regardless of `min_session_length`.

### 4.2 `finetune_learning_rate` (default: 5e-5)

Significantly lower than `pretrain_learning_rate` (2e-4). Rationale: the LoRA adapter
has useful pretraining weights that represent thousands of gradient steps. A single
Phase 2 update sees at most ~200 examples. Overshooting with a high LR would destroy
pretraining signal.

**Guidance:**
- 5e-5 is appropriate when Phase 2 fires infrequently (one session per day or less)
- If Phase 2 fires after every session and sessions are frequent (multiple per hour),
  lower to 1e-5 to prevent rapid drift
- If Phase 2 loss is not decreasing across sessions, raise to 1e-4 cautiously
- Add gradient clipping (`max_norm=0.5` for Phase 2 — tighter than Phase 1's `1.0`)

### 4.3 `fine_tuning_steps` (default: 1)

Number of full passes over the miss_log before a single optimizer step. With the
default of 1, the system does one forward pass over all miss events, accumulates
gradients, and steps once. With 2, it does two passes before stepping.

**Why this is kept low:**
- Miss events from a single session are a small, correlated batch
- Iterating multiple times over a correlated batch is equivalent to high LR overfitting
- The session's miss events will appear again in future sessions if the issue isn't
  resolved — natural curriculum handles this
- More steps per session = more risk of catastrophic forgetting of Phase 1 weights

**When to increase:** Only if you have > 200 miss events per session and L_recon is
not decreasing. Even then, 2-3 steps is the maximum before overfitting risk dominates.

### 4.4 `context_window_W` (default: 100)

The number of surrounding tokens stored alongside each segment at compression time
for use in Phase 2 teacher runs. Larger W = better teacher context = stronger
L_distill gradient, at the cost of more memory per L2/L3 entry.

**Current status:** `context_window_W` is stored in the `FullMissEvent` but not
used by `finetune.py` (see Section 3.1). This parameter only matters once the full
L_distill is implemented in Phase 2.

**Recommended value when activated:** 100 tokens (current default) is adequate for
most conversational segments. For technical or code content where wider context
strongly determines meaning, consider 200.

---

## 5. Per-session observability

### 5.1 What to log per session end

Extend `run_finetune()` to return and log the following:

```python
{
    "session_id": str,
    "miss_events_collected": int,    # raw MissEvent count
    "miss_events_reconciled": int,   # FullMissEvent count after reconciliation
    "reconciliation_drop_rate": float,  # (collected - reconciled) / collected
    "l2_miss_fraction": float,       # events.miss_level == "l2" / total
    "l3_miss_fraction": float,
    "avg_loss": float,
    "loss_by_tier": {"l2": float, "l3": float},
    "skipped": bool,
    "skip_reason": str | None,
}
```

The **reconciliation drop rate** is a valuable signal: if > 30% of miss events are
being dropped, the session is hitting L3 capacity limits and you are losing training
signal. Raise `l3_capacity` or reduce `l3_activation_threshold`.

### 5.2 Cross-session metrics (track across multiple sessions)

These reveal whether fine-tuning is improving or degrading the adapter over time:

**Phase 2 avg_loss per session** — should decrease across sessions with similar
content (adapter specializing), or stay flat when content changes domains. A
sustained upward trend means catastrophic forgetting is occurring.

**Reconstruction pass rate trend** — tracked at session end via `/stats`.
Should increase across sessions (improving compression quality). If it decreases
consistently, the Phase 2 update is hurting the reconstructor.

**Hit rate at turn 10 per session** — measure L1 hit rate at conversation turn 10
across consecutive sessions. A rising trend indicates the adapter is learning to
allocate higher importance to conversation-type content. Flat or falling means the
fine-tuning is not translating to better admission decisions.

**L2/L3 miss ratio** — `l2_miss_fraction` across sessions. If L3 misses are dominant
(the 4x compressed tier is failing), the L2→L3 compression step needs attention.
If L2 misses are dominant, the L1→L2 step is the bottleneck.

### 5.3 Structured session log format

Add a structured JSON log line at the end of every session, regardless of whether
fine-tuning fires:

```python
import json, logging
logger = logging.getLogger("training.finetune")

# In run_finetune(), always emit:
logger.info(json.dumps({
    "event": "phase2_result",
    "session_id": session_id,
    "timestamp": time.time(),
    **result_dict,
}))
```

This log is parseable for trend analysis even without a metrics backend.

---

## 6. Multi-session dynamics

### 6.1 Specialization vs. generalization

After Phase 1 pretraining, the adapter has a broad, domain-general representation.
Each Phase 2 update nudges it toward the current session's domain:

- **Frequent sessions on the same domain:** adapter specializes strongly, hit rates
  improve, miss rates fall. Good.
- **Alternating sessions across very different domains:** adapter oscillates, never
  specializes. May perform worse than the pretrained baseline. Watch for avg_loss
  trending upward across sessions.
- **Long gap between sessions:** adapter state is preserved from last session.
  First session after a gap may have higher miss rates than expected.

### 6.2 Catastrophic forgetting (deferred mitigation)

The current MVP has no forgetting protection. The risk: after 50+ sessions of
fine-tuning on session-specific content, the LoRA adapter may have drifted so far
from its pretrained state that its general-purpose compression quality degrades.

**How to detect it:**
1. After every 10 sessions, evaluate on a held-out test set from Phase 1 pretraining
   data (same documents used in Phase 1 — the adapter should still perform well on these)
2. If L_distill on Phase 1 test set rises significantly (> 0.05 above Phase 1
   converged baseline), forgetting is occurring

**Current mitigation (conservative):** Keep a copy of the Phase 1 checkpoint.
If forgetting is detected, re-initialize from Phase 1 and resume Phase 2 with a
lower LR or fewer steps per session.

**Future mitigation (EWC):** Elastic Weight Consolidation on the LoRA parameters.
Penalize changes to weights that were important during Phase 1. This is deferred
from the current MVP but is the correct long-term solution. The LoRA architecture
makes EWC lightweight — only the low-rank adapter parameters are subject to the
Fisher penalty, not the full frozen backbone.

### 6.3 Checkpoint management across sessions

The current code overwrites `checkpoints/compressor` and `checkpoints/reconstructor`
after every Phase 2 update. For multi-session experiments:

```bash
# Before each session, snapshot the current checkpoint:
cp -r checkpoints/ checkpoints_backup_session_N/
```

Or add session-stamped checkpoint saving to `run_finetune()`:

```python
from pathlib import Path
import time

stamp = int(time.time())
compressor.model.save_pretrained(f"checkpoints/compressor_session_{stamp}")
reconstructor.model.save_pretrained(f"checkpoints/reconstructor_session_{stamp}")
# Also overwrite the "latest" checkpoint:
compressor.model.save_pretrained("checkpoints/compressor")
reconstructor.model.save_pretrained("checkpoints/reconstructor")
```

This lets you roll back to any session's state if forgetting is detected.

---

## 7. Phase 3 preview: mid-inference fine-tuning

The architecture document specifies a Phase 3 where the fine-tuning step fires
**mid-session**, after accumulating N misses, rather than at session end.

**When to consider it:** Measure L1 hit rate as a function of turn number within
a session. If hit rate degrades significantly after turn 20-30 despite Phase 2
updates across sessions, end-of-session updates are too slow — the domain shifts
within the session faster than cross-session updates can track.

**What changes:**
- `min_session_length` repurposes to an intra-session batch threshold
  (e.g., "fire after 20 misses mid-session, not just at the end")
- Optimizer state must live in GPU memory alongside inference
- Backward pass must be scheduled without blocking inference
  (run asynchronously in a separate thread or after a generation completes)

**What doesn't change:** LoRA adapter architecture, frozen backbone, training
objectives, miss log format. The infrastructure is already designed for this.

**Practical cost:** One backward pass over ~20 miss events adds approximately
the same compute as one forward pass over a 400-token sequence. On a 24GB GPU,
this is feasible without latency impact if scheduled between generation calls.

---

## 8. Evaluating fine-tuning quality

### 8.1 Compression round-trip test (per-session)

Before and after Phase 2, run the same 10 held-out segments through
compress → reconstruct. Compare fingerprint similarity and confidence scores:

```bash
python -c "
from utils.config import Config
from compression.compressor import Compressor
from compression.reconstructor import Reconstructor
from semantic.fingerprinter import Fingerprinter
import torch

cfg = Config.load()
comp = Compressor(cfg, 'cpu')
recon = Reconstructor(cfg, 'cpu')
fp = Fingerprinter('cpu')
recon.set_fingerprinter(fp)

# Load a few held-out token embed tensors saved from Phase 1 test set
# For each: compress -> reconstruct -> measure fingerprint_sim
# Should be >= 0.85 post-convergence; watch for drops after Phase 2
"
```

### 8.2 Domain specificity test

Run two test conversations after 10 sessions:
1. A conversation in the **same domain** as the fine-tuning sessions
2. A conversation in a **different domain**

Compare hit rates at turn 10. The fine-tuning is working correctly if:
- Same-domain hit rate is higher than the pretrained baseline
- Different-domain hit rate is not significantly lower than the pretrained baseline

If different-domain hit rate drops by > 10%, forgetting is occurring.

### 8.3 The "conversation recall" test

This is the ground-truth behavioral test:

1. Start a session, discuss topic A for 20 turns (let it reach L2/L3)
2. Discuss an unrelated topic B for 10 turns
3. Ask a specific factual question about topic A

If the system can answer correctly, the CE preserved enough semantic content for
retrieval and reconstruction to work. Track success rate across sessions.

---

## 9. Quick reference: fine-tuning configuration

```yaml
# config.yaml — Phase 2 relevant parameters

# When Phase 2 fires
min_session_length: 20         # minimum FullMissEvents before gradient step
fine_tuning_steps: 1           # passes over miss_log before optimizer.step()

# Optimizer
finetune_learning_rate: 0.00005  # 4x lower than pretrain_learning_rate

# Teacher context (used when full L_distill is implemented)
context_window_W: 100          # surrounding tokens stored per miss event

# Capacity (affects reconciliation drop rate)
l3_capacity: 1000              # increase if reconciliation_drop_rate > 0.3
```

### Alert thresholds for Phase 2

| Metric | Warning | Action |
|---|---|---|
| `reconciliation_drop_rate` | > 0.3 | Increase `l3_capacity` |
| `avg_loss` rising across 5+ sessions | Any upward trend | Check for catastrophic forgetting; consider LR reduction |
| `l3_miss_fraction` > 0.8 | > 80% of misses are L3 | L2→L3 compression quality issue; retrain L2→L3 path |
| Phase 1 test L_distill rise | > 0.05 above baseline | Catastrophic forgetting; restore from Phase 1 checkpoint |
| `fine_tuning_steps` increasing without loss improvement | Plateau | Try LR increase or more data (accumulate misses across sessions) |
