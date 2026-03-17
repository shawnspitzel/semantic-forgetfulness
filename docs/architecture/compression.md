# Compressor Module

> The compressor's sole responsibility is reducing the dimensionality of a segment's representation when the cache controller fires an eviction event. It produces concept embedding (CE) tensors — structured dense vectors in the base LLM's token embedding space — at two compression levels: mid-compression for L1→L2 demotion and high-compression for L2→L3 demotion. It is a single unified module called with different target compression counts. It performs no importance scoring, no segmentation, and no sanity anchor extraction — those are the cache controller's and inference loop's concern.

---

## Role

The compressor is invoked by the cache controller in exactly two situations:

| Event | Input | Output |
|---|---|---|
| L1 eviction | Token embeddings of the evicted segment | L2 CE tensor (`C_L2` slots) |
| L2 eviction | L2 CE tensor of the evicted segment | L3 CE tensor (`C_L3` slots) |

Both calls go through the same module with the same weights. The only difference is the `target_c` argument, which controls the number of output CE slots and therefore the compression ratio.

---

## Interface

```
compress(
  input_embeddings: Tensor[N, d],   # token embeddings (L1→L2) or CE tensor (L2→L3)
  target_c:         int             # number of output CE slots: C_L2 or C_L3
) → Tensor[target_c, d]            # output CE tensor in LLM token embedding space
```

Both input types are sequences of `d`-dimensional vectors already in the base LLM's embedding space. For L1→L2, `N` is the segment length in tokens (~20). For L2→L3, `N` is `C_L2` — the CE count of the L2 entry being evicted. The transformer backbone processes both identically.

---

## Architecture

The compressor follows the CompLLM design: a **frozen base LLM with a LoRA adapter and a single linear projection layer**.

```
input_embeddings: Tensor[N, d]
      ↓
[ frozen LLM backbone + LoRA adapter ]   ← only LoRA weights are trained
      ↓  (processes [input_embeddings || EOS × target_c])
hidden states at EOS positions: Tensor[target_c, d_hidden]
      ↓
[ linear projection ]                    ← trained
      ↓
CE tensor: Tensor[target_c, d]
```

**Mechanism:** `target_c` learned EOS tokens are appended to the input embedding sequence. After the forward pass, the hidden states at those EOS positions are collected and projected into CE space via the linear layer. The number of EOS tokens directly controls the compression ratio — appending more EOS tokens yields more CE slots (less compression); fewer EOS tokens yield fewer CE slots (more compression).

**Frozen backbone:** The base LLM weights are not updated. Only the LoRA adapter and linear projection are trained. This preserves the base model's capabilities and ensures CEs remain in a compatible embedding space.

**L2→L3 input handling:** For L2→L3, the full input to the forward pass is constructed as `concat(L2_CE_tensor, embed(EOS_ids × target_c))` — the L2 CE tensor is used directly as the prefix, and the EOS tokens are still fetched from the frozen embedding table by their token IDs. The transformer receives a single flat sequence of `d`-dimensional vectors regardless of origin; the L2 CE slots and EOS slots are indistinguishable to the backbone.

---

## CE Structure

Every CE tensor produced by the compressor has a fixed internal layout:

```
CE[0     : E    ] → entity anchors region    (E slots, one per key entity)
CE[E     : E+B  ] → boundary region          (B slots: first sentence, last sentence)
CE[E+B   : C    ] → semantic content region  (remaining slots)
```

Where `C` is `target_c`, `E = max_entities` (a fixed hyperparameter, shared with the reconstructor), and `B = 2` (one slot per boundary sentence, fixed, shared with the reconstructor). Both `E` and `B` are named constants — the CE layout is identical for every segment regardless of how many entities were actually found. When a segment has fewer than `max_entities` entities, the remaining entity slots are zero-padded. This ensures the reconstructor can address every region by fixed offset without needing a per-segment header.

The LoRA learns to route semantic information into the correct regions through the joint training objective with the reconstructor. The structured layout is what allows the reconstructor to read each region deliberately rather than treating the CE as a flat vector.

**Across compression levels:** `E` and `B` are constant across `C_L2` and `C_L3` — they represent the same segment. Only the semantic content region shrinks as `C` decreases. At `C_L3`, the semantic content region has fewer slots than at `C_L2`; the compressor learns to distill it further.

---

## Training

The compressor is trained jointly with the reconstructor as an **autoencoder pair**. Three objectives apply simultaneously:

### Objective 1 — Hidden Activation Distillation
*(from CompLLM; L1→L2 only)*

Smooth-L1 loss between the base LLM's hidden activations when processing the original token sequence versus when processing the CE-prepended sequence. This ensures CEs live in the correct latent space for direct injection into the LLM forward pass.

```
L_distill = SmoothL1(activations_with_original_tokens, activations_with_CE_prepended)
```

Activations are measured at the answer segment layers (mid-to-late depth), normalized by layer activation standard deviation.

**This objective applies only to L1→L2 training.** For L2→L3, there is no original token sequence available at training time — only the L2 CE. `L_distill` is omitted for the L2→L3 training path; `L_recon` and `L_inject` provide sufficient signal there.

### Objective 2 — Reconstruction Fidelity

The reconstructor attempts to decompress the CE tensor back toward the original token embeddings. Loss is cosine distance between the reconstructor's output CE and the original segment's token embeddings.

```
L_recon = cosine_distance(reconstructor(CE), original_token_embeddings)
```

**Training target for both compression levels:** the reconstruction target is always the original L1 token embeddings of the segment — stored in the training corpus alongside the L2 CE. For L2→L3 training, the original token embeddings are retrieved from the training corpus by segment ID, not derived from the L2 CE. This trains the compressor to preserve semantic content recoverable all the way back to the original token sequence, not just to the previous compression stage.

The entity and boundary regions are supervised by the reconstructor's constraint checks (entity cosine similarity and boundary region initialization).

### Objective 3 — Injection Quality

An explicit term ensuring the full structured CE remains valid as soft tokens in the LLM forward pass. Measured as task performance degradation when the CE is used in place of the original tokens on held-out QA examples.

```
L_inject = task_loss(LLM with CE prepended) - task_loss(LLM with original tokens)
```

### Cache-Miss Fine-Tuning (End-of-Session)

At session end, miss events logged by the cache controller are used to fine-tune the LoRA adapter if the session produced at least `min_session_length` miss events. This is the core Semantic Forgetfulness training signal: misses indicate the compressor over-compressed content that was subsequently needed.

Training objectives applied depend on the miss level — see pretraining.md for the full fine-tuning loop specification. No weight updates occur during inference; the compressor is updated only at session boundaries.

---

## Compression Ratios

| Level | Symbol | Interpretation | Default (empirical) |
|---|---|---|---|
| L1→L2 | `C_L2` | CE slots per ~20-token segment | TBD (CompLLM baseline: 2x → 10 slots) |
| L2→L3 | `C_L3` | CE slots per L2 entry | TBD (target: 4x total from original → 5 slots) |

Both are hyperparameters tuned per model family and must satisfy `C_L2 > C_L3 > E + B` — ensuring the semantic content region has positive width at both compression levels.

---

## What the Compressor Does Not Do

- **Does not segment** — segments are sentence-boundary aligned at L1 admission by the inference loop
- **Does not compute importance scores** — attention concentration scores are computed by the inference loop at encoding time
- **Does not extract sanity anchors** — boundary sentences, entities, and semantic fingerprints are computed at L1 admission by the cache controller and passed in at call time
- **Does not reconstruct** — reconstruction from CEs is the reconstructor's responsibility
- **Does not update importance scores** — these are read-only from the compressor's perspective

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Single unified module | Same weights for L1→L2 and L2→L3, `target_c` varies | Both operations are embedding sequence → shorter embedding sequence; no architectural seam needed |
| Frozen backbone | Only LoRA and linear projection trained | Preserves base model capabilities; ensures CE space compatibility; allows compressor swapping without LLM retraining |
| EOS token mechanism | Append `target_c` EOS tokens; collect hidden states at those positions | Direct from CompLLM; linear scaling; compression ratio is a single integer argument |
| Structured CE layout | Entity / boundary / semantic regions | Gives reconstructor deliberate read targets; entity and boundary regions are stable anchors across compression levels |
| L2→L3 input | L2 CE tensor passed as `input_embeddings` directly | CEs are already in LLM embedding space; same forward pass, no separate architecture needed |
| No mid-inference weight updates | End-of-session LoRA fine-tuning only | Avoids latency penalty during serving; cache-miss signal is sparse and delayed — session-boundary processing is appropriate |
