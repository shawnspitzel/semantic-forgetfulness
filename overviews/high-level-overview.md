# Semantic Forgetfulness

## 1. The Core Idea

Current LLMs treat their context window as a flat, uniform resource. As context grows long, attention dilutes and early-but-important information gets drowned out — this is **context rot**. Semantic Forgetfulness proposes treating LLM memory as a **hierarchical cache (L1/L2/L3)**, where content is allocated by *semantic importance* rather than recency. The model learns to actively compress low-priority content into lossy "gists" (forgetting), while keeping high-priority content in fast, accessible storage. Critically, **forgetting is a first-class, trainable, goal-directed operation** — not a side effect of limited memory.

---

## 2. Novelty

> *"This is the first architecture that treats forgetting as a trainable, goal-directed operation rather than a passive consequence of context limits."*

Most prior work (CompLLM, FastKV, Pichay et al. 2026) treats compression as a static pre-processing step. This architecture is a **dynamic, self-correcting memory system** that learns its own importance model from deployment feedback via cache-miss signals. The L1 hit rate is the central training objective — clean, interpretable, and grounded.

---

## 3. High-Level Architecture

### Three Cache Levels
- **L1 (Active Context):** Full semantic richness. Raw tokens, current query, high-priority retrieved content. What the LLM actually sees.
- **L2 (Semantic Working Set):** Moderately compressed segment representations. Recently or frequently accessed content lives here.
- **L3 (Long-Term Gist Store):** Aggressively compressed. Only the "gist" survives.

### Core Loop (Inference)
1. Query arrives → search L1 first
2. L1 miss → search L2 (soft miss, moderate cost)
3. L2 miss → search L3 (hard miss, reconstruction required)
4. Reconstruction from gist → validate → promote to L2
5. Repeated L2 hits → promote to L1
6. All miss events logged as training signal

### Two Kinds of Forgetting
- **Passive forgetting:** Lossy compression. Retain the gist, discard details. Handled by the compressor module. (similar to CompLLM's "Concept Embeddings")
- **Active forgetting / suppression:** Deliberate demotion of content that is *counterproductive* to the current task — even if technically available. A qualitatively different operation. Maps to retrieval inhibition in neuroscience. (insert reference)

---

## 4. Training Strategy (Open Questions Remain)

### Pre-Training Approaches
1. **Supervised:** Teach the model what L1/L2/L3-worthy content looks like via labeled corpora (more structured and reliable, but then data sourcing becomes an issue).
2. **Self-supervised:** Model learns importance discernment from task performance signals alone. This can lead to some interesting developments, like the model figuring out that surprise, relatability, or surprise are usual signals for importance, mimicking real brain activity.
3. **Distillation:** A teacher model with full context generates targets; student model learns to match with compressed context. (Most immediately feasible — see CompLLM.)

### Core Training Objective
**Maximize L1 cache hit rate.** A high hit rate signals the model has correctly assessed importance at encoding time. By definition, strong L1 allocation implies correct L3 allocation (whatever wasn't important enough for L1/L2).

### Cache-Miss Training Signal
Cache misses during deployment (a query requires L3 content, meaning importance was misjudged at encoding) are logged and fed back as a training signal to refine the compressor and allocation policy. This is the primary mechanism for continuous learning.

### Open Question — Where Does Learning Live?
- **Weight updates at inference:** Expensive, risky, but persistent learning.
- **Cache state updates only:** Cheap, safe, but ephemeral (resets between sessions).
- *This is unresolved and needs a design decision before implementation.*

---

## 5. The Reconstruction Problem (Highest Risk Area)

When an L3 cache miss occurs, a **reconstructor module** reads the compressed gist and attempts to recover semantic meaning — potentially using surrounding context or source material as anchors.

### Problem
- If the reconstructor hallucinates, corrupted content propagates up to L1 — the highest-priority cache. Silent, confident misinformation is the worst failure mode.
- Reconstruction from source material is more reliable but potentially slow (latency cost TBD).

### Mitigation Directions
- Reconstructor should output **uncertainty-flagged representations** — system knows when a reconstruction is shaky before promoting it.
- Conservative L3 allocation — be slow to demote to L3; prefer L2 when uncertain.
- Structural "sanity anchors" alongside gists for post-reconstruction verification.
- Frame L3 as **lossy by design**, not recoverable by default. Some things are meant to be forgotten.

---

## 6. Neuroscience Grounding

The cache analogy is useful but surface-level. These two mechanisms offer deeper, more principled inspiration:

### Hippocampal-Neocortical Consolidation → Reconstruction / Replay
The hippocampus holds fast, lossy episodic traces. Important memories are later *replayed* and consolidated into rich neocortical long-term storage. The reconstructor is a learned replay mechanism. This is well-studied and gives the architecture biological plausibility.

### Synaptic Tagging and Capture → L1 Promotion Signal
Synapses active during salient events get "tagged" and preferentially capture plasticity resources. This is the biological analog of L1 priority. What makes something tag-worthy: **novelty, reward signal, prediction error, emotional valence.** These map directly to computable importance signals (surprisal, gradient saliency, etc.).

### Active Forgetting / Retrieval Inhibition → Suppression Mechanism
The brain doesn't just passively forget — it *actively suppresses* memories that interfere with current goals. This is mediated by inhibitory interneurons. The architectural analog is a suppression mechanism that demotes content not just because it's old, but because it's *counterproductive to the current task*. No current LLM memory system does this, food for thought.

Perhaps the ideal set-up is a Compression component coupled with a Supression component. Compression handles **passive forgetting**. Suppression handles **active forgetting**. 

---

## 7. Potential Benchmarks / Experiments

- **Needle-in-a-Haystack:** Tests whether critical early information survives into L1. Direct test of the core claim.
- **Long-Document QA (e.g., QuALITY, NarrativeQA):** Measures task performance degradation vs. compression ratio.
- **Multi-Turn Dialogue Coherence:** Tests whether L2/L3 promotion correctly surfaces relevant prior turns.
- **Cache Hit Rate Tracking:** Log L1/L2/L3 hit rates over a session and measure correlation with task accuracy.
- **Ablation — Active vs. Passive Forgetting:** Compare suppression-enabled vs. compression-only variants.
- **Hallucination Rate Post-Reconstruction:** Measure how often L3 reconstruction introduces factual errors before and after mitigation strategies.

---

## 8. Immediate Next Steps / Priorities

1. **Resolve the learning locus question** — weight updates vs. cache state updates at inference. This is a foundational design decision.
2. **Design the reconstructor** with uncertainty flagging as a first-class output, not an afterthought.
3. **Prototype the compressor** on a small open-weights model using distillation (closest to CompLLM — fastest path to results).
4. **Define L3 allocation conservatism policy** — what threshold triggers demotion to L3 vs. L2?
5. **Survey neuroscience literature** on synaptic tagging and active forgetting for more precise architectural inspiration.
6. **Frame the paper's central claim** around goal-directed forgetting as a trainable operation — not the cache hierarchy, which is the mechanism not the idea.

---

## 9. Open Questions

- What is the right compression ratio at each level? (L1→L2, L2→L3)
- At what context length does context rot become severe enough to justify this overhead? (Empirical threshold needed)
- Can surprisal at encoding time serve as a reliable L1 priority signal without being too expensive?
- Is active suppression best implemented as a separate module, or can it emerge from the same compressor trained with the right objective?
- How do we handle cross-session memory? L3 persistence across conversations is a different engineering problem.
- How does this interact with RAG (Retrieval-Augmented Generation)? Potential overlap or synergy.

## 10. Potential Hyperparameters
- Promotion / Eviction Leniance (promote after 1 cache miss? after 2? 10? Can tweak to find the optimal value).
- Cache Size
- Conservity (How conservative do we want to be with L3 demotions?)

---

