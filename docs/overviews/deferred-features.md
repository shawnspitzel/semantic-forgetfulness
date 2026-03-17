# Phase 2 — Deferred Features

Features discussed and intentionally deferred. Not in scope for the initial research contribution or MVP prototype.

---

## Suppression Module (Active Forgetting)

**What it is:** A component that deliberately demotes content that is *counterproductive to the current task* — not because it's unimportant, but because it's actively harmful to have in context (contradictory claims, outdated instructions, misleading premises).

**Why deferred:**
- Architecturally the least defined component in the system
- Training data is hard: "counterproductive" is context-dependent and session-dependent
- Contradictory information is often *useful* in agentic settings (e.g., surfacing conflicting sources helps an agent detect hallucinations or resolve ambiguity). A suppression module that demotes contradictions could actively hurt agent performance in these cases.
- The line between "helpful contradiction" and "harmful contradiction" is a hard open research question — getting this wrong is worse than not having suppression at all

**Potential training data directions (for when we revisit):**
- Retrospective ablation: remove segments, measure accuracy change; segments whose removal improves accuracy = suppression targets
- Synthetic contradiction injection into clean QA datasets
- TruthfulQA-style misleading context pairs

**Prerequisite:** Phase 1 compressor must be working and validated first. Suppression should be added as an ablation on top of a stable baseline.

---

## Cross-Session L3 Persistence

**What it is:** Persisting L3 gists and the entity graph across sessions, enabling long-term episodic memory.

**Why deferred:** Different engineering problem from intra-session memory. Raises data governance/privacy questions. Out of scope for the research paper's contribution, which is intra-session hierarchical allocation. Pichay et al. 2026 identifies this as "the remaining frontier."

---

## Online Weight Updates at Inference

**What it is:** Updating the LoRA adapter weights during a live serving session based on cache-miss signals, enabling session-specific adaptation in real time.

**Why deferred:** Complex serving infrastructure (training + inference in the same process), latency impact, instability risk. Requires resolving the offline learning baseline first before the marginal value of online adaptation can be measured.

---

## GNN Retriever for L3

**What it is:** A learned Graph Neural Network retriever that uses the entity co-occurrence graph in L3 for associative multi-hop retrieval, beyond nearest-neighbor ANN search.

**Why deferred:** The entity graph is built and available, but training a GNN on top of it requires a second training loop. ANN search alone is sufficient for the MVP. The GNN is a natural extension once the L3 architecture is stable.

---

## RL-Based Cache-Miss Gradient

**What it is:** Framing the allocation policy as a stochastic policy and training it via REINFORCE or policy gradient methods on the L1 hit rate reward signal.

**Why deferred:** High instability, complex infrastructure, not necessary given that retrospective supervised labels (behavioral cloning from cache-miss logs) are simpler and likely sufficient. Revisit only if supervised fine-tuning on cache-miss labels fails to generalize.

---

## vLLM / PagedAttention Integration

**What it is:** Replacing HuggingFace `past_key_values` with vLLM's PagedAttention for production-scale KV cache management.

**Why deferred:** Requires forking vLLM internals. HuggingFace is sufficient for research prototype. Relevant only at production deployment scale.
