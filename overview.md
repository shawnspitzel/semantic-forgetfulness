# Semantic Forgetfulness: A Hierarchical Memory Architecture for LLMs

## 1. Problem and Intuition

Modern large language models (LLMs) treat the context window as a flat, uniform resource. As sequences grow long, models often exhibit **context rot**: performance degrades, earlier but semantically crucial information is under‑used, and recent or superficial details dominate. This effect has been observed in practical evaluations of long‑context behavior and user reports of models “losing the thread” over time. [web:19]

Recent work reframes the context window as an **L1 cache** and builds additional memory levels around it. For example, Pichay et al. (2026) propose a four‑level memory hierarchy for LLM workloads, with an L1 generation window, L2 working set, L3 compressed history, and L4 persistent storage, managed via demand paging and pinning strategies. [web:11][web:23] While this greatly improves throughput and cost, the compression and eviction policies remain largely heuristic (e.g., recency, frequency), not deeply **semantic**.

This note proposes **semantic forgetfulness**: an explicit, learnable mechanism that (a) compresses low‑priority context segments into compact, lossy representations and (b) promotes high‑priority content into small, fast caches. The goal is to organize the effective context by **importance**, not just by recency or raw length, thereby mitigating context rot.

---

## 2. High‑Level Idea

**Goal:** Given a long context, maintain an L1 “semantic working set” that always contains the most task‑relevant information, while demoting less relevant content into lossy compressed forms at lower memory levels (L2/L3).

Core behaviors:

- **Segment long context.** Split a long input (e.g., 10k tokens) into fixed‑length segments.
- **Lossy semantic compression.** For each segment, compute a compressed representation (e.g., ~20 “summary tokens” for ~200 raw tokens).
- **Hierarchical cache.**
  - **L1:** Raw tokens and a small number of high‑priority summary tokens.
  - **L2:** Compressed segment representations for recently or frequently useful content.
  - **L3:** More aggressively compressed, cross‑session or rarely used semantic memories.
- **Query‑time retrieval.** On each query, search L1 → L2 → L3 for relevant content. When only compressed content is available (cache miss), reconstruct the needed semantics via a learned decompressor and update the hierarchy based on success/failure.

This is related to segment‑wise soft compression (e.g., CompLLM) and token‑selective KV‑cache methods (e.g., FastKV), but here compression and placement policies are explicitly trained for **semantic importance** and **cache hit‑rate**, not just efficiency. [web:21][web:26][web:17]

---

## 3. Architecture Sketch

We assume a base transformer LLM and add three main components around it: a segmenter/compressor, a hierarchical cache, and retrieval/reconstruction logic.

### 3.1 Segmenter and Compressor

**Segmenter.**  
Split the input token sequence into segments \(S_i\) of length \(L\) (e.g., 256 tokens).

**Compressor.**  
For each segment \(S_i\), produce a compressed representation \(C_i\) of length \(k \ll L\) (e.g., 16–32 pseudo‑tokens):

- A small transformer encoder or pooling module ingests \(S_i\) and outputs \(k\) learned “summary tokens.”
- These summary tokens are compatible with the base model’s embedding/hidden state space, so they can be inserted into the context as if they were regular tokens.

Design considerations (inspired by CompLLM):

- **Linear cost in input length.** Compression should scale linearly with the number of segments. [web:21][web:26]
- **Reusable segments.** Compressed representations \(C_i\) are cached and reused for future queries over the same underlying content. [web:21][web:26]

In contrast to pure autoencoding, the compressor is later trained to preserve **task performance**, not just reconstruct text.

### 3.2 Hierarchical Cache (L1 / L2 / L3)

Conceptually, we treat LLM memory like a cache hierarchy, similar to CPU/OS designs and recent LLM memory‑hierarchy work. [web:11][web:23]

- **L1: Active Context**
  - Contents:
    - Current query tokens.
    - A sliding window of recent raw tokens.
    - A small budget of compressed “summary tokens” retrieved from lower levels.
  - This is the actual context the base LLM sees during forward passes.

- **L2: Semantic Working Set**
  - Contents:
    - Compressed segment representations \(C_i\) for segments that have been recently or frequently used.
  - Behavior:
    - “Demand paging”: when the model needs information from a segment, its \(C_i\) is brought into L1 (as pseudo‑tokens or after reconstruction).
    - Promotion on use; demotion/eviction under pressure, similar to working‑set and pinning strategies. [web:11]

- **L3: Long‑Term Compressed Memory**
  - Contents:
    - More aggressively compressed, higher‑level summaries (e.g., episode‑level vectors, cross‑session summaries).
  - Implementation:
    - Stored in an embedding index or vector database.
    - Retrieved via similarity search against the current query or hidden state.

As in traditional memory systems, performance depends on **L1/L2 hit rates**, not just raw L1 size. [web:11][web:23]

### 3.3 Retrieval and Reconstruction

For a new query \(Q\):

1. **Initial L1 Build**
   - Include:
     - Query tokens.
     - A small window of most recent raw tokens.
     - Any obviously relevant compressed tokens (e.g., those attached to the last few dialogue turns).

2. **Semantic Retrieval from L2/L3**
   - Use \(Q\) (or its hidden representation from the base model) as a query into L2 and L3:
     - Compute similarity between \(Q\)’s representation and \(C_i\) (L2) / higher‑level vectors (L3).
     - Select top‑\(m\) compressed representations to bring into L1.

3. **Cache Hits**
   - For each selected compressed representation:
     - **Option A (no reconstruction):** insert compressed tokens \(C_i\) directly into L1 as pseudo‑tokens.
     - **Option B (with reconstruction):** run a decompressor to map \(C_i\) back into a sequence of “reconstructed tokens” or richer hidden‑state segments, and insert those into L1.

4. **Cache Misses**
   - If none of the available representations suffice to answer correctly (as judged during training or via a teacher model), treat this as a **cache miss**:
     - Record which segments were relevant but poorly compressed or not retrieved.
     - Use this signal to update the compressor, decompressor, and retrieval policy.

---

## 4. Training Objectives

The system can be trained in two main stages, on top of a base language model.

### 4.1 Compression Pretraining / Distillation

Given a frozen or teacher LLM:

- For each training example:
  1. Run the teacher with full, uncompressed context to obtain target outputs.
  2. Run the student architecture with:
     - Segmented input.
     - Earlier segments replaced by compressed representations \(C_i\) (or reconstructed variants).
  3. Optimize the compressor (and decompressor, if used) to minimize:
     - **Task loss:** cross‑entropy between student and teacher outputs (or ground truth).
     - Optional: reconstruction loss between teacher’s hidden states and the student’s reconstructed hidden states for the same segments.

This is analogous to CompLLM, which shows that 2× compression can preserve or even improve long‑context QA performance compared to uncompressed baselines. [web:21][web:26]

### 4.2 Semantic Forgetfulness and Cache‑Miss Training

To incorporate **intentional forgetfulness** and cache hierarchies:

- **Simulate Cache Pressure**
  - During training, enforce budgets on L1 and L2:
    - Evict segments from L1 to L2 and from L2 to L3 based on simple policies (e.g., recency) or randomization.
    - Restrict the model to using only compressed representations for many past segments.

- **Cache‑Aware Loss**
  - Task loss:
    - Standard language modeling / QA objective over the model’s outputs.
  - Cache‑miss penalties:
    - Define a metric when relevant information was only accessible through deeper levels or when retrieval failed.
    - Penalize configurations that require frequent, expensive reconstruction from L3 or fail on tasks because of over‑compression.
  - Regularization:
    - Encourage compressed representations to be low‑dimensional and sparsely informative to prevent trivial “memorize everything” solutions.

Over time, the compressor and cache policy learn what information to keep at higher levels and what can be safely demoted or discarded, optimizing for **functional performance** rather than token‑level fidelity.

---

## 5. Minimal Prototype Plan

A practical path to an initial prototype:

1. **Base Setup**
   - Choose an open‑weights LLM.
   - Implement a segmenter with fixed segment length (e.g., 256 tokens).

2. **Compressor Module**
   - Implement a small transformer (or MLP + pooling) compressor that turns each segment into 16–32 pseudo‑tokens.
   - At inference, for long contexts:
     - Keep the most recent segment raw.
     - Replace earlier segments with their compressed tokens in the prompt.

3. **Training / Evaluation**
   - Use long‑context benchmarks (e.g., long‑document QA, needle‑in‑a‑haystack tasks).
   - Compare:
     - Full‑context baselines.
     - Segment‑compressed models with various compression ratios.
   - Optimize compressor parameters to preserve or improve accuracy at reduced effective context, similar to CompLLM. [web:21][web:26]

4. **Add Simple L1/L2 Hierarchy**
   - Maintain an in‑memory L2 store of compressed segments.
   - Use recency/frequency heuristics to decide which compressed segments are:
     - Kept in L1 as pseudo‑tokens.
     - Stored only in L2.
   - Log “miss” events where the model needed information from evicted segments.

5. **Iterate toward Learned Policies**
   - Replace heuristics with:
     - A learned retrieval head for selecting which \(C_i\) to bring into L1 on each query.
     - A cache‑miss‑aware objective that encourages better semantic compression and placement.

---

## 6. Future Directions

Several extensions follow naturally from this architecture:

- **Empirical Study of Compression vs Context Rot**
  - Measure how varying compression ratios (e.g., 2×, 4×, 8×) affect performance and robustness to long prompts.
  - Compare naive long‑context models vs models with semantic compression for their tendency to “forget” early content. [web:19][web:21][web:26]

- **Importance Signals**
  - Explore different signals for segment priority:
    - Surprisal (model’s own prediction error).
    - Gradient‑based saliency on downstream tasks.
    - User‑provided markers (e.g., “pin this,” “this is the objective”).
    - Task‑specific labels (e.g., constraints vs examples).

- **Neuroscience‑Inspired Dynamics**
  - Incorporate ideas from synaptic consolidation and neuromodulated learning:
    - Different timescales for updates at each level (fast L1, slow L2/L3).
    - Stabilization of “important” compressed representations, akin to protected synapses. [web:13][web:18][web:20]

- **System‑Level Integration**
  - Combine this semantic hierarchy with existing demand‑paging controllers and memory‑hierarchy proposals for LLMs:
    - Use OS‑style paging to manage raw tokens.
    - Use learned semantic compression to decide *what* is worth keeping close to the model. [web:11][web:23]

This **semantic forgetfulness** framework aims to turn context management into a first‑class, trainable component of LLM architectures, aligning memory usage with what actually matters for downstream performance rather than simply expanding context windows.

