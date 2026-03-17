# Problem Answers

## General: What LLM Will We Use?
Recommendation: Use Llama 3.1/3.2 (8B or 70B) or Qwen3 (4B or 7B) as the base model.

## P0.3 — Unit of Compression

**Recommendation: Semantic segments of approximately 20 tokens, dynamically aligned to sentence or clause boundaries.**

CompLLM (arXiv:2509.19228) is the most direct empirical evidence here. Their system uses a fixed segment size of **S=20 tokens** as the unit of compression, producing C=S/compression_rate concept embeddings per segment. This choice is validated across NarrativeQA, SQuAD, RACE, QuAIL, and LOFT (up to 128k tokens).

**Avoid fixed-token-count chunks without boundary alignment.** The Redis context management guide and the Context Rot paper both implicitly support this: semantic coherence at boundaries matters because attention degrades when segments straddle topic transitions.

**Practical recommendation:** Use 20-token segments as the default, with a lightweight sentence-boundary detector (e.g., spaCy sentence tokenizer or newline/period heuristics) to snap segment boundaries to the nearest sentence end within ±5 tokens.

---


## P2.1 — What Architecture Does the Compressor Use?

**Recommendation: A frozen base LLM + LoRA module + single linear projection layer, following CompLLM exactly.**

CompLLM (arXiv:2509.19228) provides the strongest empirical basis for a compressor architecture. Their design:

- Takes the base LLM itself as the encoder backbone, adding a LoRA adapter to steer its representations
- Appends S/C EOS tokens to the input segment; the hidden states at these positions become the C concept embeddings
- A single linear layer maps from hidden dimension to concept embedding space
- The base LLM weights are **not updated** — only the LoRA and linear layer are trained
- Training objective: hidden activation distillation (Smooth-L1 loss matching answer segment representations across layers, normalized by layer activation standard deviation)

The advantage of this design over alternatives:

- **Cross-attention encoder** (option from problems.md): more complex, requires designing a separate encoder architecture, slower to train.
- **Autoencoder on reconstruction loss**: reconstruction-only objectives tend to over-retain surface form rather than semantic content. Distillation against the base LLM's own activation space is better aligned.
- **Small generative LLM summarizer**: token sequences as gists are 10-100x larger than dense concept embeddings for equivalent semantic coverage; they introduce generation latency in the compression path.
- **Linear projection of mean-pooled states**: loses positional and structural information that the LoRA-augmented per-position EOS token approach preserves.

**Caveat:** The LoRA-on-frozen-LLM approach requires that the base LLM used for compression is the same family as the one used for generation (since concept embeddings must live in the correct token embedding space). This is a constraint, not a bug — it enforces architectural coherence.

---

## P2.2 — What is the Gist Format?

**Recommendation: Dense continuous vectors (concept embeddings) in the base LLM's token embedding space, not token sequences and not sparse vectors.**

CompLLM (arXiv:2509.19228) directly settles this. Their concept embeddings are:

- Continuous vectors in the same latent space as token embeddings
- Fixed dimensionality (matching the LLM's hidden dimension)
- Directly prependable to the LLM's input without any fine-tuning of the LLM
- Produced at a rate of 1 CE per 2 tokens (at 2x compression)

Why not token sequences? A natural-language summary is interpretable and debuggable, but it reintroduces the token budget problem — a 20-token segment compressed to a 10-token summary saves only 50% and requires autoregressive generation in the compression path. Concept embeddings achieve the same semantic coverage with a single forward pass and no vocabulary constraints.

Why not sparse vectors? Sparse representations require a compatible retrieval index (inverted index rather than HNSW) and cannot be directly injected into the LLM forward pass. They offer interpretability but impose a decoder step that dense vectors avoid.

**For L3 specifically:** the gist stored in L3 is a sequence of C concept embeddings (where C = segment_length / compression_rate). The vector index stores these CE sequences. Retrieval returns a CE sequence that can be directly prepended to the LLM's input at L2/L1 promotion time. This is the cleanest possible interface between storage and consumption.

---

## P2.4 — Does the Compressor Need Joint Training with the Base LLM?

**Recommendation: No. Train the compressor independently with the base LLM frozen. This is sufficient and substantially simpler.**

CompLLM (arXiv:2509.19228) explicitly validates this design choice. Their architecture:
- Leaves the base LLM weights completely unchanged
- Trains only the LoRA adapter and linear projection
- Uses hidden activation distillation to ensure the concept embeddings are compatible with the frozen LLM's forward pass

This achieves **4x TTFT speedup and 50% KV cache reduction** without any modification to the base model. The implicit worry — that the LLM won't know how to use compressed representations — is addressed by the training objective: by matching the LLM's own intermediate activations during training, the compressor learns to produce representations the frozen LLM can already consume.

**Why not joint training?** Joint training requires backpropagating through the full LLM, which is prohibitively expensive, and risks destabilizing the base model's capabilities. It also eliminates the ability to swap the compressor for different compression ratios without retraining the LLM.

**Practical implication:** Start with the frozen LLM + LoRA compressor design. If ablations show that the frozen LLM cannot adequately consume reconstructed gists at the L3 promotion step (a real risk for aggressive compression ratios), add a lightweight adapter layer trained on reconstruction quality as a second phase — still without modifying core LLM weights.

---

## P9.2 — What is the Baseline?

**Recommendation: Use four baselines in this order of priority.**

| Baseline | Purpose | Source |
|---|---|---|
| **Full context (oracle)** | Upper bound on quality; measures how much the compression costs | Any long-context benchmark |
| **CompLLM (2x compression)** | Primary comparison for novelty; the most capable prior art for soft compression | arXiv:2509.19228 |
| **FastKV (token saliency + KV pruning)** | Comparison for KV-level compression; tests whether importance-weighted KV dropping is competitive | arXiv:2502.01068 |
| **Naive truncation (oldest-first)** | Lower bound; baseline that any reasonable system must beat | Standard practice |
| **StreamingLLM** | Tests against a system that handles infinite context via attention sink + sliding window | Xiao et al. 2023 |

**The comparison to CompLLM is the most important.** Semantic Forgetfulness must demonstrate either: (a) better task performance at the same compression ratio, or (b) better compression ratio at the same task performance. The claim of novelty — that cache-miss-driven allocation and multi-tier promotion improve over static per-segment compression — must be supported by this comparison.

**Note on LLMLingua-2:** CompLLM itself compares to LLMLingua-2 and outperforms it. Including LLMLingua-2 in the baseline suite is optional but adds depth to the compression landscape.

**Evaluation datasets:** Use LOFT (up to 128k tokens), NarrativeQA, and LongMemEval (conversational QA). LongMemEval is particularly important because the Context Rot paper (Chroma Research 2026) shows it exhibits "significantly higher performance on focused prompts compared to full prompts," meaning it directly tests the core capability Semantic Forgetfulness is designed to improve.

---

## P1.1 — What Signal Quantifies Importance at Encoding Time?

**Recommendation: Use attention weight concentration (low entropy = high importance) as the primary signal, averaged across heads and measured from the Token-Selective Propagation layer (mid-depth), following FastKV.**

The literature supports a clear hierarchy of signals:

**1. Attention weight concentration (recommended):** FastKV (arXiv:2502.01068) builds its entire compression architecture on this signal. Tokens that receive high cumulative attention weight from subsequent context tokens (specifically, from recent "window tokens") are retained; low-weight tokens are pruned. Their TSP layer identifies the middle of the network as the optimal location to measure this — early layers haven't built full context representations, late layers have already committed to predictions. FastKV achieves within 1% accuracy at 10-20% KV retention using this signal alone.

**2. Attention entropy as a derived measure:** For any position, attention entropy measures how spread or concentrated incoming attention is. Low entropy (concentrated attention from many heads onto this token) is a strong signal that the token is load-bearing for the current context. This is directly computable from the attention weight matrix without any additional model components. IBM's attention mechanism reference confirms that attention weights of 1.0 indicate "100% of attention" — concentration is the meaningful signal, not raw weight values.

**3. Gradient saliency (Hou & Castanon 2023, arXiv:2308.05219) — do not use at inference:** Gradient-based saliency (and the decoding layer variant proposed in that paper) requires a backward pass, which doubles inference compute and is incompatible with the latency budget. The paper is useful for offline analysis and distillation target construction, but not as a real-time importance signal.

**4. Surprisal — do not use as primary signal (see P1.2).**

**5. Learned importance head:** viable as a second-phase addition once the attention-based signal has been validated, but introduces training complexity upfront that is unnecessary given the quality of the attention signal.

**Practical implementation:** At each segment boundary, compute the mean attention weight each token received from all subsequent tokens in the segment, averaged over all heads. Tokens in the top-K% by this score are flagged as high-importance. The segment-level importance score is the mean token importance within the segment. This feeds directly into the eviction policy (P4.2).

---

## P1.2 — Is Surprisal a Reliable Importance Proxy?

**Recommendation: No. Do not use surprisal as a primary importance signal. Use it only as a weak auxiliary feature if at all.**

The theoretical problem is real and the literature does not resolve it in surprisal's favor:

**The fundamental ambiguity:** High surprisal means the model did not predict the token — which could indicate (a) genuinely novel, important information or (b) noise, error, or domain shift. Low surprisal means the token was predictable — which could indicate (a) redundant filler or (b) well-established context that is foundational to the conversation. Neither direction cleanly separates important from unimportant content.

**What the literature suggests instead:** FastKV's attention-based signal is empirically superior because it measures *how much the model actually used* a token when processing subsequent context, not how surprising the token was when first encountered. Attention concentration is a retrospective signal (did this token matter for what came after?) rather than a predictive signal (is this token unusual?). The retrospective framing is more directly aligned with the question "should we keep this in cache?"

**The one case where surprisal has value:** Identifying *anomalies* — tokens with extremely high surprisal that persist across multiple layers and multiple heads may signal genuine out-of-distribution content that the model is struggling with. This could be used as a flag for "do not compress this segment aggressively" rather than as a positive importance score. In that limited role, surprisal is a useful defensive signal.

**Recommendation:** Include surprisal as a secondary feature in the importance scoring function (alongside attention concentration), but weight it inversely — very high surprisal should *reduce* compression aggressiveness, not increase importance score. Do not use it as a standalone importance predictor.

---

## P5.4 — How Do We Handle Catastrophic Forgetting if Weight Updates Happen at Inference?

**Recommendation: Use Elastic Weight Consolidation (EWC) for the compressor and importance head if online weight updates are required. Prefer LoRA-restricted updates to limit the parameter surface exposed to online learning.**

Kirkpatrick et al. 2017 (PNAS) introduced EWC as a direct solution to catastrophic forgetting in continual learning. The method:

- Computes the Fisher information matrix for each parameter after training on a prior task, approximating the posterior over weights
- Adds a quadratic penalty to the loss for any subsequent task: `L = L_new + λ * Σ_i F_i(θ_i - θ*_i)^2`
- Parameters with high Fisher information (important for prior tasks) are penalized heavily for deviating from their prior values; unimportant parameters are free to adapt
- This selectively "slows down learning on the weights important for those tasks" (per the abstract)

**Why EWC is the right choice here:**
- It does not require storing full previous datasets (unlike experience replay), which matters for privacy in a session-memory system
- The penalty is computed once per "task boundary" (in our case, after each session's offline training batch) and applied cheaply during online updates
- It is compatible with LoRA: apply EWC to the LoRA adapter weights only, not the full LLM (which remains frozen regardless)

**Practical hybrid recommendation:**
1. **At inference:** No weight updates. Only cache state changes (content is moved between L1/L2/L3). This is the safest path and avoids EWC complexity during serving.
2. **Post-session (offline):** Batch the cache-miss signals from one or more sessions, run a LoRA fine-tuning step on the compressor, apply EWC penalty using Fisher information computed from the initialization checkpoint.
3. **Fisher information approximation:** Use the diagonal approximation (not full matrix), which scales as O(parameters) not O(parameters²). For a LoRA adapter on a 4B model with rank 16, this is computationally trivial.

**Do not do online weight updates at inference time** unless the latency budget explicitly allows a backward pass per token. The risk/reward ratio is unfavorable: the compressor will overfit to the current session's patterns and fail on any distribution shift.

---

## P3.3 — What Are "Sanity Anchors" Concretely?

**Recommendation: Store three lightweight artifacts alongside each L3 gist: (1) the first and last sentence of the original segment, (2) extracted named entities and numbers, and (3) a sentence-level embedding from a frozen encoder.**

The sanity anchor must satisfy a tight constraint: it must be small enough not to undermine the compression ratio, but rich enough to catch hallucination-level reconstruction failures. Based on information retrieval and NLP practice:

**1. Boundary sentences (first + last sentence of segment):** These serve as positional anchors — the compressor can hallucinate content in the middle of a segment, but is far less likely to produce plausible boundary sentences that are factually wrong. Storage cost: ~20-40 tokens per segment, which is offset by the 10 CEs saved by 2x compression. This is the single highest-value anchor.

**2. Named entities, dates, numbers:** Extracted by a lightweight NER tagger (e.g., spaCy's small model, ~12MB). Factual anchors (proper nouns, numerical values) are the class of content most likely to be hallucinated during reconstruction and most likely to cause downstream errors when wrong. Storing the entity surface forms costs ~5-15 tokens per segment for typical conversational content.

**3. Frozen sentence embedding:** A 768-dim vector from a small frozen encoder (e.g., `all-MiniLM-L6-v2`, 22M parameters) captures the segment's semantic fingerprint. At reconstruction time, encode the reconstructed text and compute cosine similarity. If similarity < threshold (empirically ~0.7-0.8 based on sentence transformers literature), flag the reconstruction as uncertain and do not promote to L1. This is the most principled check but requires one encoder forward pass per reconstruction.

**Validation logic:** A reconstruction passes sanity if: (a) cosine similarity to stored embedding ≥ threshold AND (b) all stored entities are present in the reconstruction (or have been explicitly removed with a reason). Failing either condition keeps the content in L2 with an uncertainty flag rather than promoting to L1.

**What to avoid:** Key-phrase extraction alone (option from problems.md) is insufficient — key phrases can be reproduced verbatim by a hallucinating reconstructor while the surrounding claims are fabricated. The embedding similarity check catches this.

---

## P8.2 — Does the System Require a Specific Model Architecture?

**Recommendation: Require HuggingFace Transformers as the inference interface for prototyping. Add vLLM support once the core system is validated. Do not attempt to support closed-API models.**

The KV cache access requirement is the binding constraint. The two viable options are:

**1. HuggingFace Transformers `generate()` with `past_key_values`:** This is the standard interface for KV cache access in research. The `past_key_values` argument returns the full KV cache after each forward pass, allowing inspection, modification, and selective retention. All Llama, Qwen, Mistral, and Gemma model families support this interface. **This is the right choice for the prototype.**

**2. vLLM with PagedAttention:** vLLM's PagedAttention physically pages KV cache blocks in GPU memory and exposes block-level management. This is closer to the Pichay et al. 2026 demand paging model. However, modifying vLLM's PagedAttention internals requires forking vLLM and is substantially more engineering work. **Use vLLM in a second phase for production-scale experiments.**

**What does NOT work:**
- OpenAI API, Anthropic API, Gemini API: no KV cache access
- llama.cpp: KV cache is accessible but Python API is more limited; viable for quantized models if GPU memory is constrained
- ONNX Runtime: no dynamic KV manipulation support

**The Pichay et al. 2026 architecture** (transparent HTTP proxy at the message array level) is instructive: they achieve demand paging without modifying the model at all, by operating at the text/message level rather than the KV cache level. This is a valid fallback design if KV-level access proves too complex — the compressor can operate on token sequences and inject concept embeddings as prepended context rather than modifying the KV cache directly. CompLLM demonstrates this works well.

**Practical recommendation:** Start with HuggingFace `past_key_values` on a Qwen3-4B or Llama 3.2-3B model on a single A100 or H100. This gives full access to KV internals, fast iteration, and direct compatibility with the CompLLM codebase (which also uses HuggingFace).

---

## P10.2 — At What Context Length Does Context Rot Justify This Overhead?

**Recommendation: The intervention threshold is approximately 4,000–8,000 tokens for conversational workloads and 16,000–32,000 tokens for document retrieval workloads. Below these thresholds, standard attention is adequate.**

The Context Rot paper (Chroma Research 2026) provides the relevant empirical data:

**Conversational / repeated-access tasks:** Performance degrades beginning around **500–750 words (~650–1,000 tokens)** in repetition and working memory tasks. Claude Opus 4 begins task refusal around 2,500 words (~3,300 tokens). For practical multi-turn agentic conversations, the Redis data is more relevant: "15+ turns = ~30,000 tokens" with "7x latency increase at 15,000 words of context." This suggests the useful quality cliff is around **10,000–20,000 tokens** for conversational workloads.

**Document retrieval / NIAH tasks:** Performance is more stable, remaining adequate until mid-to-high input lengths. The Context Rot paper shows effects becoming pronounced with lower needle-question similarity pairs — meaning less retrievable content degrades faster. Practical threshold: ~**32,000–64,000 tokens**.

**Key insight from Context Rot:** "Structural coherence paradoxically harms performance — models perform better on shuffled haystacks than logically structured ones." This means the degradation is not simply a function of length but of how the context is organized. Semantic Forgetfulness, by selectively retaining high-importance content and demoting low-importance content, directly addresses this: it changes what is present in L1, not just how much.

**Recommendation for the paper:** Report the activation threshold as a hyperparameter with default value of **8,000 tokens for L2 engagement** and **32,000 tokens for L3 engagement**, empirically tuned against task type. Below 8k tokens, the compression and retrieval overhead (even at O(N)) is not worth the quality tradeoff.

**Caveat:** These thresholds are model-dependent. Llama 3.1's effective context length under standard attention has known degradation above 32k tokens even within its 128k window. Measure empirically on the chosen base model before finalizing activation thresholds.

---

## P4.2 — What is the Exact Eviction Policy?

**Recommendation: Importance-weighted LRU, where importance is measured by attention concentration score and the recency component uses a time-decayed access count. Evict the segment with the lowest combined score.**

The scoring function:

```
eviction_score(s) = alpha * importance(s) + (1 - alpha) * recency(s)
```

Where:
- `importance(s)` = normalized attention concentration score computed at encoding time (from FastKV's saliency signal)
- `recency(s)` = exponentially decayed time since last access: `exp(-lambda * (t_now - t_last_access))`
- `alpha` = importance weight (recommend starting at 0.6, tune as hyperparameter)
- Evict the segment `argmin_s eviction_score(s)`

**Why not pure LRU?** Pichay et al. 2026 use FIFO by turn age as their baseline and observe a 25% fault rate on Read evictions. LRU without importance weighting fails because frequently-accessed but low-importance content (e.g., repeated tool schema definitions) blocks genuinely important but rarely-accessed content from surviving. Pichay's fault-driven pinning (content that causes a page fault gets permanently pinned) is a coarser version of importance weighting.

**Why not pure importance threshold?** A fixed importance threshold fails in dynamic sessions where the importance distribution shifts — early segments that scored high at encoding time may become irrelevant as the conversation evolves.

**Why not LFU?** Access frequency over a long session strongly biases toward early content regardless of current relevance. The exponential decay on recency corrects for this.

**The training objective aligns with this policy:** The cache-miss rate training signal directly penalizes incorrect evictions (evicting content that was subsequently needed). The gradient flows back through the importance score, which trains the compressor to produce higher scores for content that will be accessed again. This closes the loop between the eviction policy and the training objective.

**Pichay et al. 2026's inverted cost model is important here:** Unlike physical memory where keeping data resident is free, keeping tokens in L1 costs compute every turn (quadratic attention). This means the eviction policy should be *more* aggressive than classical LRU would suggest — erring toward eviction and tolerating higher fault rates is often optimal.

---

---

## P0.4 — How is L3 Different from RAG?

**Answer: L3 is a demand-paged, miss-driven, LLM-native gist store — mechanically and behaviorally distinct from RAG on four axes.**

The L3 architecture document makes the distinction precise:

**1. Retrieval trigger:** RAG retrieves on every query, scanning the external corpus proactively. L3 retrieves only on an **L2 cache miss** — demand paging, not proactive search. L3 is never scanned on a turn where L1/L2 already contains the needed content. This is the OS virtual memory model, not the search engine model.

**2. Content origin:** RAG documents come from an external corpus that exists before the conversation. L3 content was *generated by the model during the current session* — compressed from the model's own prior context. The content in L3 is a lossy trace of what the model has already processed, not a reference corpus.

**3. Representation format:** RAG stores text (or text embeddings for retrieval). L3 stores **concept embedding tensors in the LLM's own token embedding space** — representations that can be prepended directly to the LLM's forward pass without decoding to text first. These are not semantic search vectors; they are soft tokens native to the specific model.

**4. Associative structure:** L3 includes an in-memory entity co-occurrence graph that enables 1-hop associative retrieval beyond nearest-neighbor lookup. RAG systems use ANN search only. The graph extension is the substrate for a learned GNN retriever — a qualitatively different retrieval paradigm.

**For the paper's framing:** L3 is best described as *personalized, ephemeral, LLM-native episodic memory* — not retrieval over external knowledge. The RAG overlap exists only at the mechanical level (vector index + similarity search); the semantics, trigger condition, and content type are different.

---

## P3.1 — What Architecture Does the Reconstructor Use?

**Answer: Anchored Progressive Reconstruction (APR) — a two-stage, context-grounded, constraint-first pipeline. Not a free generative decoder.**

The reconstructor.md document fully specifies this. The core insight is that reconstruction is a **narrowing funnel, not a generation task** — the reconstructor has known ground-truth constraints and must fill in the minimal interior, not freely produce text. Three design principles govern all decisions: hard constraints before soft generation; ground in real L2 content, not model priors; fidelity over richness.

### Layer Structure

**Layer 1 — Constraint Shell (applied unconditionally):**
- `boundary_sentences` are copied verbatim — the reconstructor never touches them
- `entities` from `sanity_anchors` are a hard inclusion list — output missing any entity is rejected and retried (up to retry budget, then fallback to boundary sentences only)
- `semantic_fingerprint` cosine check is the exit gate — output that fails is not promoted

**Layer 2 — Structured CE Format:**
Rather than a flat dense vector, the concept embedding has an explicit internal layout:
```
CE[0 : E]       → entity anchors region
CE[E : E+B]     → boundary region
CE[E+B : D]     → semantic content region
```
The reconstructor reads each region deliberately rather than decoding a flat vector. Compressor and reconstructor are **trained jointly as an autoencoder pair** with two objectives: reconstruction fidelity on held-out segments + injection quality loss (CE remains valid as soft tokens in the LLM forward pass).

**Layer 3 — Context-Grounded Expansion:**
When L2 is warm, the reconstructor queries the entity graph for L2 neighbors of the segment being reconstructed. These neighbors serve as grounding context — real text the reconstructor fills from rather than generating from model priors. If no L2 neighbors exist, falls back to boundary sentences + entity list only.

### Two-Stage Pipeline

**Stage 1 — L3→L2 (Fidelity Pass):** Produce a sparse, high-confidence skeleton. Do not elaborate beyond what is anchored. Lock boundary sentences, verify entities, fill interior from L2 grounding, emit with per-span confidence scores.

**Stage 2 — L2→L1 (Query-Conditioned Enrichment Pass):** Enrich the skeleton with content relevant to the current query. Re-apply constraint shell, score each span for query-relevance, expand high-relevance spans using L2 neighbors, leave low-relevance spans from the L2 skeleton unchanged.

The same L3 segment may be enriched differently on different retrievals depending on what was asked. This mirrors context-dependent recall.

---

## P3.2 — How Does the Reconstructor Produce Uncertainty Estimates?

**Answer: Deterministic pass/fail via semantic fingerprint check, with per-span confidence scores as observability instrumentation. The fingerprint check gates promotion; confidence scores inform debugging and training.**

The reconstructor.md architecture makes the uncertainty mechanism explicit:

**The gate (hard):** `cosine(output_embedding, semantic_fingerprint) ≥ θ`. This is the semantic fingerprint stored in `sanity_anchors`. If the reconstructed text's MiniLM embedding is too far from the stored fingerprint, the output is rejected and not promoted. This is not probabilistic — it is a deterministic quality check.

**Per-span confidence scores (soft):** The reconstructor emits confidence scores alongside every reconstruction for every span. These are observability instrumentation only — they do not gate promotion. They are used for:
- Debug traces (which spans was the reconstructor uncertain about)
- Training signal (high-confidence wrong spans = hallucination hotspots; low-confidence right spans = over-caution)
- Offline aggregate analysis of systematic compressor failure modes

**Fallback path:** Content that fails the fingerprint check stays in L2 with the uncertainty flag rather than being promoted. If the retry budget is exhausted, the system falls back to boundary sentences + entity list only — a minimal but verified skeleton.

**What remains empirical:** The threshold θ for the cosine similarity check requires calibration on a held-out reconstruction set. The current recommendation from P3.3 is θ ≈ 0.7–0.8 based on sentence transformers literature, but the exact value is a hyperparameter to tune.

---

## P3.4 — What is the Acceptable Latency Budget for L3 Retrieval + Reconstruction?

**Answer: The target is 100ms end-to-end for L3 retrieval. Current design hits ~20–25ms, leaving 75ms of headroom.**

The L3 architecture document provides a concrete latency breakdown per operation:

| Step | Operation | Latency |
|------|-----------|---------|
| 3 | ANN search (HNSW, 100K entries) | ~10ms |
| 4 | Entity graph 1-hop expansion | <1ms |
| 5 | Optional CE sequence rerank | ~5ms |
| 7 | Transfer CE tensor to GPU + prepend | ~1ms |
| **Total** | | **~20–25ms** |

**Why 100ms is the budget:** The L3 path is triggered only on an L2 miss — a relatively rare event during a session. The 100ms figure is an interactive latency budget that keeps L3 retrieval below the perceptual threshold for a hard miss, consistent with web-standard P99 latency goals for user-facing applications.

**Why disk-backed storage is ruled out:** The tiered KV cache literature shows disk I/O introduces >100ms from layout transformation overhead alone — this puts any disk-backed store outside the budget regardless of SSD speed. The entire L3 index is therefore kept **in CPU DRAM** and never touches disk during a session.

**Storage backend:** HNSW vector index (hnswlib or FAISS) for ANN search + in-memory entity graph (NetworkX or custom adjacency list). Graph linking is built at write time so read-time expansion is <1ms.

---

## P2.3 — Compressor Training Objective? (Updated)

**Recommendation (updated): Two-objective joint training — hidden activation distillation (from CompLLM) plus injection quality loss, with the compressor and reconstructor trained as an autoencoder pair.**

The reconstructor architecture (reconstructor.md) resolves the previously-open question of how to combine CompLLM's distillation objective with the reconstruction requirement:

**Objective 1 — Hidden activation distillation:** Smooth-L1 loss matching the base LLM's answer-segment activations (from CompLLM). This ensures concept embeddings live in the correct latent space for direct injection.

**Objective 2 — Reconstruction fidelity:** The compressor is trained jointly with the reconstructor as an autoencoder pair. Loss on held-out segment reconstruction. This trains the compressor to fill the structured CE regions (entity / boundary / semantic) in ways the reconstructor can decode.

**Objective 3 — Injection quality:** An explicit term ensuring the structured CE remains valid as soft tokens in the LLM forward pass — neither objective cannibalizes the other.

**The cache-miss signal** (the core Semantic Forgetfulness objective) is still an offline signal fed back post-session as a LoRA fine-tuning step. The two training objectives above apply during initial supervised training; the cache-miss gradient refines the allocation policy over deployment.

**The previously-flagged conflict** — that reconstruction objectives and distillation objectives might push the CE format in different directions — is resolved by the structured CE layout. The entity/boundary/semantic regions split the CE space so that distillation quality (the semantic content region) and decodability (the entity + boundary regions) are not competing for the same latent dimensions.

---

## P7.1 — Is L3 Persistent Across Sessions?

**Answer: No. L3 is explicitly scoped to a single session. Cross-session persistence is out of scope for the initial research contribution.**

The L3 architecture document states this directly in the "What L3 Is Not" section:

> "Not cross-session — L3 is scoped to a single session. Cross-session persistence is explicitly out of scope for the initial research contribution (see Pichay et al. 2026 for framing)."

**For the paper:** Pichay et al. 2026 identify cross-session memory as "the remaining frontier" — cite this to bound the contribution. The paper's claim is about intra-session hierarchical allocation, not persistent episodic memory.

**P7.2 (privacy)** is therefore also deferred — if L3 does not persist, there is no persistent storage of user conversation content, and data governance concerns are out of scope for the prototype.

---

## P8.1 — How Does This Interact with RAG Pipelines?

**Answer: L3 is not a RAG system and does not overlap with RAG pipelines — they operate at different scopes and with different content. They can co-exist without conflict; L3 does not replace RAG.**

The L3 architecture document makes the distinction explicit. RAG retrieves from a pre-existing external corpus on every query. L3 retrieves from the model's own compressed session history only on an L2 cache miss. A system with both RAG and Semantic Forgetfulness uses:
- RAG for external knowledge that was never in the conversation
- L3 for prior conversation content that has been compressed out of L1/L2

The two systems are complementary, not competing. L3 gists are in the LLM's native concept embedding space and cannot be used in a standard RAG pipeline (which expects text or generic semantic embeddings). There is no architectural merge to worry about.

**If a future system wants to unify them:** L3 could serve as a "session RAG" layer sitting alongside a traditional RAG corpus. The retrieval trigger would remain different (miss-driven for L3, query-driven for external RAG). This is out of scope for the initial paper but is a natural extension.

---

## Problems Deferred (Require Experiments)

The following problems cannot be answered from the current literature or architecture documents and require empirical work:

**P0.1 — What exactly is L1 in transformer terms?** The choice between "subset of KV cache entries," "separate attention stream," and "context window tokens with no modification" is fundamentally a prototype design decision that cannot be resolved from literature alone. CompLLM's approach (inject concept embeddings as prepended context) and Pichay's approach (message-level proxy) represent two different valid answers. Start with CompLLM's approach (L1 = token context window with injected concept embeddings), then ablate.

**P1.3 — How is task relevance computed for active suppression?** No literature directly addresses dynamic task-relevance scoring for suppression. Requires experiment design.

**P3.2 — Calibrated uncertainty threshold (θ)?** The fingerprint cosine check architecture is settled, but the threshold value requires calibration on a held-out reconstruction set. Empirical work needed.

**P4.1 — How are cache sizes set and bounded?** L2 and L3 size are free hyperparameters with no strong prior from literature. Requires ablation.

**P4.3 — What triggers promotion from L3 to L2 to L1?** Fault-driven pinning (Pichay) is a reasonable starting point, but the exact fault count threshold requires empirical tuning.

**P4.4 — How is importance score updated over time?** No existing system addresses retrospective importance re-scoring within a session. Novel contribution territory; requires design + experiment.

**P5.1 — How is the cache-miss signal converted into a gradient?** Behavioral cloning from an oracle is the most tractable starting point (P5.3 addresses this), but whether it generalizes to the online setting requires an experiment.

**P5.2 — Where does learning live?** The recommendation (offline weight updates only) is a strong prior, but the value of session-specific online adaptation is an open empirical question.

**P5.3 — What data is needed for distillation?** Dataset size and domain requirements for the compressor's training set require empirical measurement of sample efficiency.

**P5.5 — Pre-training strategy?** The pretraining document is currently empty. This is unresolved: whether the initial allocation policy should be random, heuristic (attention-based), or trained before any deployment feedback is received requires a design decision.

**P6.1–P6.3 — Suppression module:** All suppression questions are deferred. Start with compressor-only; add suppression as an ablation in a second phase.

**P9.1 — How is L1 hit rate defined rigorously?** Requires a formal operational definition before training can begin. Propose: "the fraction of query-relevant segments present in L1 at query time, measured by cosine similarity ≥ θ between the query embedding and each L1 segment's concept embedding."

**P9.3 — Minimum viable prototype?** Recommended first experiment: single-session long-document QA. Use a 50k-token document. Measure accuracy of Semantic Forgetfulness (CompLLM compressor + importance-weighted LRU eviction) vs. full context oracle vs. naive truncation on NarrativeQA. No online learning in this first prototype — just validate that the tiered cache with attention-based importance scoring outperforms truncation.

**P10.1 — Compression ratios at each level?** CompLLM shows 2x is safe; 4x is plausible but requires validation. L2→L3 aggressive compression ratios are unexplored. Requires ablation across compression rates.
