# Inference Loop

> This document specifies the per-turn inference flow — how user input is processed, how importance scores are computed, how context is constructed for the LLM forward pass, and how the session starts and ends. The cache controller, compressor, and reconstructor are orchestrated here; this document describes the sequence of calls, not the internals of those components.

---

## Implementation Interface

**HuggingFace Transformers with `past_key_values` and `output_attentions=True`.**

All inference runs through the HuggingFace `generate()` interface. `past_key_values` is the KV cache — the stored Key and Value matrices from all prior turns, which allow the model to attend over prior context without recomputing it from scratch each turn. `output_attentions=True` returns per-layer, per-head attention weight matrices on every forward pass, which is required for importance score computation.

The base LLM is loaded once at session start and held in GPU VRAM. The LoRA adapter is attached on top. Neither is modified during inference.

---

## Session Lifecycle

### Session Start

```
1. Load frozen base LLM → GPU VRAM
2. Load LoRA adapter (most recently saved checkpoint) → attach to base LLM
3. Initialize cache controller (empty metadata index, empty L1/L2/L3 tier stores)
4. Initialize miss_log = []
5. Initialize conversation_history = []   # running append-only token log for session
6. Initialize entity graph (empty)
7. Set session_id
8. Begin REPL loop
```

### Per-Turn Loop

```
1.  Receive user input string
          ↓
2.  Tokenize → token sequence
          ↓
3.  Append tokens to conversation_history
          ↓
4.  Segment into ~20-token sentence-boundary-aligned chunks
    (see Segmentation)
          ↓
5.  For each new segment:
      a. Run encoding forward pass with output_attentions=True
         (see Importance Score Computation — one pass per segment, MVP)
      b. Extract importance score from TSP layer attention weights
      c. Compute semantic fingerprint via frozen MiniLM-L6-v2
      d. Cache controller admission:
           — assign tier = l1
           — if L1 at capacity: run eviction before inserting
          ↓
6.  Cache controller miss detection:
      a. Encode current query → query_vec via MiniLM
      b. Scan metadata index: flag segments where
         cosine(query_vec, fingerprint) ≥ miss_detection_theta AND tier ≠ l1
      c. For each soft miss (tier = l2):
           trigger L2 retrieval + L2→L1 enrichment pass (reconstructor)
      d. For each hard miss (tier = l3):
           trigger L3 retrieval + full reconstruction pipeline (reconstructor)
      e. Log each miss to miss_log
          ↓
7.  Rebuild KV cache from full L1 content (including any segments just promoted in step 6)
    (see KV Cache Management)
          ↓
8.  Construct context tensor:
      [ all L1 segments by source_position (raw + promoted, interleaved) | current query ]
      Note: promoted CE soft tokens from step 6 are now part of L1 and included here by source_position,
      not as a structurally separate prefix
          ↓
9.  Run generate() with output_attentions=True → response tokens
          ↓
10. Decode → print to terminal
          ↓
11. Tokenize and segment assistant response tokens
          ↓
12. Append to conversation_history
          ↓
13. For each assistant response segment:
      — extract importance score from generation attention weights
        (no separate encoding pass needed — weights are free from step 9)
      — compute fingerprint via MiniLM
      — cache controller admission
```

### Session End

Triggered by process exit, SIGTERM, or `/done` command.

```
1. Flush partial segment buffer into L1 as-is
2. Reconcile miss_log → full MissEvents (see Miss Logging)
3. If len(miss_log) >= min_session_length:
     run end-of-session fine-tuning (see pretraining.md)
4. Save updated LoRA adapter to disk
5. Discard all tier content (session-scoped, not persisted)
6. Exit
```

---

## Segmentation

Tokens are accumulated in a buffer. A boundary is triggered when:
- Buffer has ≥ 20 tokens **and** the last token ends a sentence (`.` `?` `!` `\n`)
- **Or** buffer reaches 25 tokens (hard cap — prevents unbounded buffering on run-on text)

This produces segments of 15–25 tokens, sentence-boundary aligned.

**Partial segments** (at session end or message boundary): admitted to L1 as-is. CE slot count is scaled proportionally to token count, subject to the `E + B + 1` minimum from the CE layout.

---

## Importance Score Computation

**Signal:** attention weight concentration at the TSP layer (Token-Selective Propagation, `tsp_layer_index = floor(num_layers / 2)`). Tokens that receive concentrated attention from subsequent tokens are load-bearing for the context — this is the importance signal FastKV validates as reliable for KV pruning decisions.

### For User Input Segments

A dedicated encoding forward pass is required — user tokens are not generated, so no attention weights exist yet.

```
Input:  [ current L1 token buffer (positions 0..L-1) ] + [ new segment tokens (positions L..L+S-1) ]
Pass:   frozen LLM, output_attentions=True
Extract: attention matrix A at tsp_layer_index
         A[i, j] = how much does position i attend to position j (causal: i ≥ j)

For each token t in the new segment (at position p_t in the full input):
    token_importance(t) = mean over all positions i in (p_t+1 .. L+S-1) of A[i, p_t],
                          averaged across all heads
    — i.e., the mean attention that all later segment tokens give to t

segment_importance = mean(token_importance) over the segment
segment_importance = (segment_importance - session_running_mean) / session_running_std
```

Running mean and std are maintained per-session for stable score distribution across turns.

**Directionality note:** In causal attention, position i can only attend to positions j ≤ i. `token_importance(t)` measures how much subsequent tokens in the segment attend BACK to t — i.e., how load-bearing t is for the tokens that follow it. Because segment tokens are at the END of the encoding pass input, "subsequent context" is limited to within-segment tokens that appear after t. The last token in a segment has no subsequent tokens and will always score zero by this formula — this is expected and acceptable since the last token has not yet been attended to by any new content.

**One forward pass per segment (MVP).** Batching multiple segments into one pass would reduce latency at the cost of changing the attention context each segment sees, which may degrade score quality. Marked as an optimization opportunity for post-MVP.

### For Assistant Output Segments

Attention weights are available for free from the `generate()` call. No extra forward pass. At each segment boundary (generation is complete for that segment before scoring), extract TSP layer weights and compute importance:

```
For each token t in the completed segment (at position p_t in the full generated sequence):
    token_importance(t) = mean over all positions i in (p_t+1 .. segment_end) of A[i, p_t],
                          averaged across all heads
    — same column-sum formula, applied to generation attention weights
```

The formula is structurally identical to the encoding pass case. The difference is that generation attention covers the full prior conversation context (all L1 content + prior turns), so the signal is richer for tokens earlier in the segment than for the encoding pass case where the prior context is just the current L1 buffer.

---

## KV Cache Management

**Full rebuild each turn.** Rather than surgically removing evicted segment entries from `past_key_values`, the KV cache is rebuilt from scratch from the current L1 token buffer at the start of each turn (step 7 in the per-turn loop).

This means one extra encoding forward pass per turn over all current L1 content. The tradeoff: zero positional index tracking, zero tensor surgery, and no risk of stale or misaligned KV entries. For an MVP terminal chatbot where L1 is a small set of ~20-token segments, this is acceptable. If turn latency becomes a bottleneck, surgical eviction is the first optimization candidate.

---

## Context Construction

```
context = [
    all L1 content ordered by source_position,   # raw token segments + promoted CE soft tokens,
                                                  # interleaved by original conversation position
    current query tokens
]
```

Promoted CEs (from L2/L3 reconstructions) are injected into L1 as soft tokens during the enrichment pass (reconstructor.md Stage 2, step 6) and are ordered by `source_position` alongside raw token segments. They are not a structurally separate prefix — they are L1 content, distinguished only by `is_reconstructed = true` in the metadata index.

**The CompLLM "prepend" framing** refers to the KV cache management approach: rather than inserting CE soft tokens at their historical KV positions (which would require tensor surgery), we rebuild the KV cache from the full L1 buffer each turn. The entire L1 buffer — both raw and reconstructed content — is treated as a flat sequence ordered by `source_position` and prepended to the current query as the KV rebuild input. This is the CompLLM approach.

**Note on positional encoding:** Segments that were evicted and reconstructed may have `source_position` values spread across the conversation history. When the KV cache is rebuilt each turn from L1 content ordered by `source_position`, the LLM sees those positions in their original relative order but without the evicted segments in between. This is an accepted MVP tradeoff — ablation against a full-context baseline is planned as part of the benchmark suite.

---

## Miss Logging

The cache controller logs a lightweight event per miss:

```
{segment_id, miss_type, query_vec, timestamp}
```

The end-of-session fine-tuning step requires richer events. At session end, a reconciliation pass assembles full MissEvents from the raw log, using `conversation_history` as the source for context windows:

```
for event in raw_miss_log:
    # Total misses have no surviving tier data — skip them
    if event.miss_type == "total":
        continue

    segment = tier_store.lookup(event.segment_id)
    if segment is None:
        continue  # defensive guard: segment evicted between miss and reconciliation

    if event.miss_type == "soft":
        # L2 miss: segment_input is the original token embeddings
        # Retrieve from conversation_history by source_position
        segment_input = conversation_history[
            segment.source_position : segment.source_position + segment.original_length
        ]
        miss_level = "l2"
    else:
        # L3 miss: segment_input is the L2 CE tensor at time of L2→L3 demotion
        # This must be stored on the segment at eviction time (see cache-control.md)
        segment_input = segment.l2_ce_at_demotion
        miss_level = "l3"

    full_event = MissEvent(
        segment_input    = segment_input,
        context_window   = conversation_history[
                               segment.source_position - W :
                               segment.source_position + W
                           ],
        ce_produced      = segment.ce_tensor,
        miss_level       = miss_level,
        session_position = segment.source_position
    )
```

**Note:** L3 miss reconciliation requires that `l2_ce_at_demotion` is stored on the segment entry at the time of L2→L3 eviction (when the compressor runs on the L2 CE to produce the L3 CE). The cache controller must preserve this field on the L3Entry — it cannot be reconstructed at session end. This is an implicit requirement on the L3Entry struct (currently not in L3.md) that needs to be added.

**conversation_history** is a running append-only log of all raw tokens seen this session, maintained separately from the tier stores. It enables context window retrieval at session end without requiring original tokens to survive the compression pipeline. Memory overhead is proportional to session length — acceptable for the terminal chatbot MVP.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| HuggingFace `past_key_values` | Primary inference interface | Full KV access; all major open-weight model families; CompLLM compatible |
| `output_attentions=True` | Always on | Required for TSP importance scoring; no other path to attention weights via HF |
| One encoding pass per user segment | Simple, accurate | Batching is an optimization — deferred post-MVP |
| Full KV rebuild per turn | Chosen over surgical eviction | Zero index tracking, zero tensor surgery; acceptable latency for MVP |
| Promoted CEs ordered by source_position in L1 | Not a structural prefix separate from L1 | Consistent with L1.md and reconstructor.md; interleaved with raw tokens by original position |
| KV rebuild = full L1 buffer prepended to query | Prepend, not historical KV insertion | CompLLM-validated; avoids tensor surgery; positional encoding impact is a known ablation item |
| Conversation history buffer | Separate from tier stores | Enables miss log reconciliation without original tokens surviving compression |
| Single-threaded | One query at a time | No concurrency edge cases; consistent with cache-control.md |

---

## Open Questions

| ID | Question |
|----|----------|
| I1 | TSP layer index: model-family-dependent; must be empirically calibrated per base model before first training run |
| I2 | Positional encoding impact: evicted-and-reconstructed segments retain their original source_position but the tokens between them are gone from the KV rebuild; ablation against full-context baseline required |
| I3 | Encoding pass batching: post-MVP optimization opportunity — batch user segments per message for lower latency |
| I4 | Partial segment `target_c` scaling rule below 20 tokens: proportional scaling subject to `E + B + 1` minimum |
| I5 | L3Entry must store `l2_ce_at_demotion` and `original_length` fields — required by miss log reconciliation; these fields are not currently in L3.md and must be added |
