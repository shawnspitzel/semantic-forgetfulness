# Cache Controller

> The cache controller is the orchestration layer of the Semantic Forgetfulness hierarchy. It owns no model weights and performs no generation. It is the single authority on tier membership — deciding what content lives in L1, L2, and L3 at every point in a session. Every other component (compressor, reconstructor, entity graph) operates at the cache controller's direction.

---

## Role

The cache controller makes three classes of decisions:

1. **Admission** — when new content enters the system, assign it a tier and compute its fingerprint
2. **Eviction** — when a tier is full, nominate and eventually demote the lowest-value segment
3. **Promotion** — when a miss recurs enough times, surface the segment to a higher tier

It maintains the metadata index, enforces the warm-start protocol, detects misses, and logs training signal. It does **not** update importance scores — that is handled in the inference loop (see `inference.md`).

---

## Metadata Index

The cache controller maintains a CPU-resident metadata index over all segments across all tiers:

```
SegmentMetadata {
  id:                    UUID
  tier:                  "l1" | "l2" | "l3"
  importance_score:      float            # set at encoding time; updated by inference loop
  last_accessed:         timestamp
  source_position:       int              # original token position in conversation
  session_id:            str
  is_reconstructed:      bool             # l1 only: false = raw input, true = promoted from l2
  semantic_fingerprint:  Tensor[768]      # frozen MiniLM-L6-v2 embedding, computed at admission
  fault_count:           int              # miss events this segment has triggered (promotion counter)
}
```

The metadata index is the control plane. Actual content (token sequences in L1, skeleton text in L2, CE tensors in L3) lives in tier-appropriate stores. The cache controller never transforms content directly — it instructs the compressor or reconstructor to do so and writes the result into the target tier.

---

## Admission

When a new raw segment enters from user input, assistant output, or tool results:

```
1. Decode segment tokens → text
       ↓
2. Run frozen MiniLM-L6-v2 on text → semantic_fingerprint: Tensor[768]
       ↓
3. Create SegmentMetadata with tier = "l1", fault_count = 0, low_score_count = 0
       ↓
4. If L1 is at capacity → run eviction (see Eviction) before inserting
       ↓
5. Insert token sequence into L1 GPU buffer
       ↓
6. Register metadata in index
```

Fingerprint computation at admission is the only MiniLM call the cache controller makes for raw input segments. Content arriving from the reconstructor already carries a verified fingerprint from the reconstruction sanity check — the cache controller reuses it directly.

---

## Miss Detection

On each incoming query turn:

```
1. Encode the query → query_vec: Tensor[768]  (frozen MiniLM)
       ↓
2. For each segment in the metadata index, compute:
   relevance(s) = cosine(query_vec, s.semantic_fingerprint)
       ↓
3. A segment is "needed" if relevance(s) ≥ θ
       ↓
4. A needed segment is a miss if its tier ≠ "l1"
```

Classify each miss:

| Miss type | Condition | Action |
|---|---|---|
| **Soft miss** | `tier = "l2"` | Trigger L2 retrieval + L2→L1 enrichment pass |
| **Hard miss** | `tier = "l3"` | Trigger L3 retrieval + full reconstruction pipeline |
| **Total miss** | absent from all tiers | Content is permanently gone; no recovery |

On any miss, increment `fault_count` on the missing segment's metadata entry. When `fault_count ≥ leniency`, trigger promotion on the next miss event (see Promotion).

---

## Eviction

Eviction fires **on admission when a tier is at capacity** — not on a schedule. On each such event:

```
eviction_score(s) = 0.6 * importance_score(s) + 0.4 * exp(-λ * (t_now - t_last_accessed(s)))
```

**Selection:**

Evict the segment with the lowest `eviction_score` in the current tier immediately. No counter or nomination accumulation — a full tier is the trigger and the threshold.

**On eviction, the target tier determines what happens to the content:**

| Evicted from | Action |
|---|---|
| L1 | Compressor produces mid-compression CE tensor (`C_L2` slots) → write as L2Entry with `origin = "l1_demotion"` |
| L2 | Compressor produces high-compression CE tensor (`C_L3` slots) → write as L3Entry |
| L3 | Permanently dropped (L3 is the floor) |

After eviction, update the evicted segment's `tier` in the metadata index.

---

## Promotion

Promotion fires when a segment's `fault_count ≥ leniency`, meaning the segment has been needed but absent from L1 enough times that surfacing it is worth the cost.

**On promotion:**

| Promoted from | Action |
|---|---|
| L2 | Reconstructor runs L2→L1 enrichment pass (refines CE toward current query) → inject CE into L1 as soft tokens |
| L3 | Reconstructor runs L3→L2 fidelity pass, then L2→L1 enrichment pass → inject CE into L1 as soft tokens |

After promotion, update `tier = "l1"` in the metadata index and reset `fault_count` to 0. If L1 is at capacity, eviction runs before the promoted segment is inserted.

---

## Warm-Start Protocol

The cache controller gates L3 demotion implicitly through `l2_capacity`. L2→L3 eviction cannot fire until L2 is full — meaning the first `l2_capacity` segments evicted from L1 always land in L2, naturally seeding it before any L3 activity begins.

This ensures the reconstructor's context-grounded expansion has a valid, high-quality grounding corpus before any L3 retrieval is attempted. L2 is always seeded from L1 demotions — the model's own active context — never from cold or random content. The conservatism of the warm-start is controlled by a single knob: a larger `l2_capacity` means more content accumulates in L2 before L3 demotion begins.

---

## Activation Thresholds

The cache controller does not engage all tiers from session start:

| Context length | Behavior |
|---|---|
| < 8,000 tokens | Standard attention; L2 and L3 inactive. No compression or eviction. |
| ≥ 8,000 tokens | L2 engagement activates. L1→L2 demotion enabled. |
| ≥ 32,000 tokens | L3 engagement activates. L2→L3 demotion enabled. |

Both thresholds are hyperparameters, tuned per model family and task type. Below 8k tokens, compression and retrieval overhead outweighs any quality benefit.

---

## Training Signal Logging

Every miss event is logged by the cache controller for offline post-session use:

```
MissEvent {
  segment_id:   UUID
  miss_type:    "soft" | "hard" | "total"
  query_vec:    Tensor[768]
  timestamp:    timestamp
}
```

These logs are batched post-session and used to fine-tune the compressor's LoRA adapter. The cache controller collects but does not process them — training happens offline.

---

## Hyperparameters

| Hyperparameter | Role | Default |
|---|---|---|
| `leniency` | Fault count before a missed segment is promoted | TBD empirically |
| `alpha` | Importance weight in eviction scoring | 0.6 |
| `lambda` | Decay rate for recency component | TBD empirically |
| `theta` | Cosine similarity threshold for miss detection | TBD empirically (~0.7–0.8) |
| `l2_capacity` | Maximum segments in L2; implicitly governs warm-start conservatism | TBD |
| `l3_capacity` | Maximum segments in L3 | TBD |
| `l2_activation_threshold` | Context length at which L2 engages | 8,000 tokens |
| `l3_activation_threshold` | Context length at which L3 engages | 32,000 tokens |

---

## Interfaces

### Inbound (cache controller receives)

| Signal | Source | Action |
|---|---|---|
| New raw segment | Inference loop | Admission |
| Reconstructed segment | Reconstructor | Admission (fingerprint reused) |
| Query vector | Inference loop | Miss detection |
| Updated importance scores | Inference loop | Metadata index update |

### Outbound (cache controller triggers)

| Trigger | Target | Condition |
|---|---|---|
| L2 retrieval + enrichment | Reconstructor | Soft miss |
| L3 retrieval + reconstruction | Reconstructor | Hard miss |
| L1→L2 compression | Compressor | L1 eviction |
| L2→L3 compression | Compressor | L2 eviction |
| Miss event log | Training signal store | Every miss |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Eviction trigger | On admission, not on schedule | Keeps eviction synchronous with the only event that changes tier capacity |
| Leniency on both transitions | Single hyperparameter governing fault count and low-score count | Prevents thrashing; one bad turn does not cause cascading tier churn |
| Fingerprint at admission | Computed once for raw segments, reused for reconstructed | Avoids redundant MiniLM calls; reconstruction already verifies the fingerprint |
| No importance score ownership | Scores updated by inference loop, not cache controller | Separation of concerns; cache controller reads scores, inference loop writes them |
| Warm-start implicit via `l2_capacity` | No separate `conservity` gate | L2 can't evict to L3 until full; capacity is the only knob needed |
| Warm-start owned by cache controller | Not the reconstructor | Reconstructor assumes L2 is warm; controller guarantees it |
| Single-threaded (MVP) | One query processed at a time | Eliminates concurrency edge cases during prototype validation |
| Training signal: collect only | Logs batched and processed offline | No backward pass at inference time; avoids latency penalty and catastrophic forgetting risk |
