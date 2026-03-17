# Pretraining & Training Strategy

> This document specifies how the compressor is trained — both initial pretraining before first deployment and the online learning loop that refines it during use.

---

## What Gets Trained

**Only the compressor's LoRA adapter and linear projection layer.**

The base LLM is frozen at all times — during pretraining, during fine-tuning, and during inference. Its weights are never touched. The reconstructor is trained jointly with the compressor as an autoencoder pair (see compression.md, reconstructor.md). No other components receive gradient updates.

---

## Phase 1: Initial Distillation Training

The compressor must produce useful concept embeddings before the first session. Without pretraining, the first session's allocations are essentially random — the miss log would be noise, and the first end-of-session fine-tuning step would corrupt the adapter rather than improve it. Pretraining establishes a working importance function before any deployment feedback arrives.

### What We're Training

Given a segment of ~20 tokens from a longer document, the compressor must produce a CE tensor such that when that CE is injected into the frozen base LLM in place of the original tokens, the LLM's intermediate activations are approximately the same as they would be with full context. In other words: the CE must be a functionally equivalent substitute for the original tokens, from the LLM's perspective.

### Training Loop

```
For each training document:
  1. Chunk document into ~20-token segments (sentence-boundary aligned)
  2. Run the full document through the frozen base LLM → record hidden states H_full at answer-segment layers
  3. For each segment:
       a. Run compressor (LoRA-augmented frozen LLM) → produce CE tensor
       b. Inject CE into frozen LLM in place of original tokens → record H_compressed
       c. Compute joint loss (see below)
       d. Backprop through LoRA adapter + linear projection only
```

### Training Objectives

Three objectives apply simultaneously (from compression.md):

**L_distill** — Hidden activation distillation (L1→L2 training only):
```
L_distill = SmoothL1(H_compressed, H_full)
```
Activations measured at mid-to-late layers, normalized by layer activation standard deviation. Ensures the CE lives in the correct latent space for direct LLM injection. Not applicable to L2→L3 training (no original tokens available at that stage; L_recon and L_inject provide the signal there).

**L_recon** — Reconstruction fidelity:
```
L_recon = cosine_distance(reconstructor(CE), original_token_embeddings)
```
Trains the compressor-reconstructor autoencoder pair. Reconstruction target is always the original L1 token embeddings — not derived from intermediate CE stages.

**L_inject** — Injection quality:
```
L_inject = task_loss(LLM with CE prepended) - task_loss(LLM with original tokens)
```
Ensures the CE remains valid as soft tokens in the LLM forward pass. Prevents L_distill and L_recon from optimizing independently at the expense of downstream task performance.

### Dataset

**Open question — requires a decision before implementation:**
- What documents? Long-document corpora (NarrativeQA, LOFT, BooksCorpus, or similar) are natural candidates given the task distribution. Task distribution should match or be broader than expected deployment use.
- What teacher model? The teacher is the frozen base LLM itself (same model family as the compressor backbone) run on the full document. No external model required — the teacher is the same frozen model the compressor is built on.
- How large? Minimum dataset size for a stable CE signal is unknown. Requires measuring sample efficiency empirically. CompLLM provides a reference point but does not report exact dataset sizes.

---

## Phase 2: End-of-Session Online Learning (MVP)

### Session Definition

For the terminal chatbot MVP, a **session = one process lifetime.** The user starts the chatbot script, converses, and exits. Session end is triggered by process exit (SIGTERM handler or explicit `/done` command). There are no persistent sessions, no cross-session state.

### Miss Log

Throughout a session, every L2 and L3 cache miss is logged to an in-memory list:

```
MissEvent {
  # For L2 misses: original token embeddings of the segment (Tensor[N, D])
  # For L3 misses: the L2 CE tensor that was input to L2→L3 compression (Tensor[C_L2, D])
  # These are different types. Do not treat them interchangeably.
  segment_input:             Tensor[N_or_C, D]

  context_window:            Tensor[W, D]   # surrounding tokens at time of original encoding
                                            # (window size W is a hyperparameter — open)
  ce_produced:               Tensor[C, D]   # the CE the compressor actually produced
  miss_level:                "l2" | "l3"    # which tier was hit — determines training objectives
  session_position:          int            # token position in conversation at encoding time
}
```

A miss event means the compressor under-estimated this segment's importance — it produced a CE that caused the segment to be demoted too aggressively, or allocated with too low an importance score, such that it was not in L1/L2 when subsequently needed.

### end-of-session Fine-Tuning Step

On session exit:

```
1. If len(miss_log) < min_session_length:
       skip update — insufficient signal, do not update weights
2. For each MissEvent in miss_log:
       if miss_level == "l2":
           # segment_input is original token embeddings
           a. Run frozen base LLM on (context_window + segment_input) → H_full
           b. Run compressor(segment_input, target_c=C_L2) → CE_new
           c. Run frozen LLM with CE_new injected → H_compressed
           d. Compute L_distill(H_compressed, H_full) + L_recon + L_inject
              ⚠ L_inject requires task supervision — see Open Questions
       if miss_level == "l3":
           # segment_input is the L2 CE tensor; original tokens no longer available
           # L_distill does not apply here (no original token sequence)
           b. Run compressor(segment_input, target_c=C_L3) → CE_new
           c. Compute L_recon + L_inject only
              ⚠ L_inject requires task supervision — see Open Questions
3. Accumulate gradients over all miss events
4. Run optimizer step on LoRA adapter + linear projection weights
   Note: no forgetting protection is applied in the MVP. This is an accepted risk — see Open Questions.
5. Save updated adapter to disk
```

**Gradient step scheduling:** One full pass over the miss_log with gradient accumulation, then a single optimizer step. For a small miss_log (at the min_session_length threshold), this means one optimizer step over ~20 examples. For a large miss_log, the same single-step structure applies — the miss_log is not re-sampled or iterated multiple times per session update. `fine_tuning_steps` is therefore a multiplier on this: how many passes over the miss_log to run before the single optimizer step. Default: 1.

### Why Missed Segments Become Training Examples

The cache miss is a retrospective label: this segment was needed but wasn't available in L1/L2, meaning the CE the compressor produced for it did not preserve enough semantic content for the system to recognize it as important. Re-running the distillation objective on specifically the missed segments trains the compressor to produce better CEs for them in future sessions — not by telling it "be more conservative," but by giving it a concrete signal of what a good CE for that segment looks like (i.e., one that matches the teacher's hidden states).

### min_session_length Hyperparameter

**Type:** int
**Default:** TBD empirically
**Interpretation:** Minimum number of cache miss events in a session before the end-of-session weight update fires.
**Rationale:** A very short session (few turns, little content demoted to L2/L3) will produce few or zero misses. Updating the LoRA adapter on 1-2 examples would be noisier than not updating at all. This threshold gates the update to sessions where enough signal was collected for a stable gradient step.
**Open question:** What value of N produces a stable update? CompLLM and related work do not address online fine-tuning dynamics at this granularity. Requires empirical measurement — start with a conservative value (e.g., 20 miss events) and tune down.

---

## Phase 3: Mid-Inference Online Learning (Post-MVP)

The architecture for mid-inference learning is identical to end-of-session — same LoRA adapter, same training objectives, same miss-log format. The only change is **when the fine-tuning step fires**: instead of on session exit, it fires after accumulating N miss events mid-session (N is a new hyperparameter).

What this enables: the compressor adapts to the current session's domain distribution within the session, not only across sessions. For long agentic sessions where domain is narrow and stable (e.g., a coding agent working in one codebase for hours), this is meaningfully different.

**The forcing function for this transition:** instrument L1 hit rate as a function of turn number within a session. If hit rate plateaus or degrades after the early warm-start window despite end-of-session updates across sessions, that is the signal to move to mid-inference updates. This measurement applies to chatbot sessions and does not require an agentic workload to evaluate.

**What changes in the serving infrastructure:**
- The serving process must hold optimizer state in GPU memory alongside the model (memory overhead — currently unquantified)
- A backward pass must be schedulable during inference (latency impact — currently unquantified)
- The min_session_length threshold repurposes to a per-inference-batch threshold

**What does not change:** LoRA adapter architecture, training objectives, miss log format, base LLM frozen status.

---

## Hyperparameters

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `min_session_length` | int | TBD | Minimum cache miss events before end-of-session update fires |
| `lora_rank` | int | 16 | LoRA adapter rank; controls trainable parameter count |
| `learning_rate` | float | TBD | Fine-tuning learning rate for end-of-session update |
| `fine_tuning_steps` | int | TBD | Number of gradient steps per end-of-session update |
| `context_window_W` | int | TBD | Token window stored alongside each miss event for teacher re-run |
| `distill_layer_range` | tuple | TBD | Which LLM layers L_distill is measured at (mid-to-late per CompLLM) |

---

## Open Questions

- **Context window W for teacher re-run (L2 misses):** At end-of-session, the teacher needs surrounding context to produce a meaningful H_full for an L2 miss event. Full conversation history would be ideal but is potentially 50k+ tokens per missed segment — expensive and may exceed GPU memory mid-update. A windowed context of width W is a practical tradeoff. Note that this makes Phase 2 teacher targets weaker than Phase 1 teacher targets (which use the full document). Phase 2 updates are therefore training against a lower-quality signal than Phase 1 pretraining. This asymmetry is acknowledged and accepted as an MVP tradeoff; the Phase 2 updates are refinements, not full retraining.
- **L_inject task supervision in Phase 2 (resolved):** The task for L_inject at end-of-session is the query that triggered the miss. Supervision target is the teacher's response (frozen LLM run with full context including the missed segment). Concretely: reconstruct context at miss time from the conversation history buffer, run teacher to get the "correct" response, run student with CE in place of the segment, compute `L_inject = loss(student_response, teacher_response)`. This requires one additional generation pass per missed segment at session end — the teacher forward pass for L_distill already runs on the same context, so this extends that pass to also generate a response.
- **Non-missed segments in the update batch (resolved):** Missed segments only for MVP. No implicit negatives. If the compressor drifts toward over-retention (treating everything as high importance), it will be visible in hit rate tracking. Revisit in Phase 2 if the drift manifests.
- **Dataset size for Phase 1:** Minimum training set size for stable initial distillation is empirically unknown.
- **Learning rate and step count for end-of-session update:** Standard fine-tuning heuristics may not apply given the small batch size and online setting. Requires measurement.
- **Catastrophic forgetting across sessions (accepted known gap):** The current design applies no forgetting protection during end-of-session updates. Multiple sessions with different domain distributions could degrade the adapter. This risk is accepted for the MVP. If empirical measurements show meaningful cross-session degradation, EWC on LoRA weights is the candidate mitigation (deferred to Phase 2).
