# Benchmarks and Baselines for Evaluating Semantic Forgetfulness

## Goal of the Experiments

This project should be evaluated as a **memory systems paper**, not
purely as a language model paper.

### The headline claim

> **Importance-aware hierarchical compression achieves task performance within 5%
> of full-context inference at 4× memory reduction, and improves further across
> deployment sessions via self-supervised learning from cache-miss signals.**

This claim has two parts. The first is a performance-efficiency tradeoff claim
(standard in compression papers). The second is the novel contribution — no prior
compression system improves from deployment.

### Supporting sub-claims (each needs its own experiment)

- **Sub-claim 1 (Hierarchy):** Allocating compression depth by learned importance
  preserves more task-critical information than uniform compression or recency-based
  truncation at the same memory budget.
- **Sub-claim 2 (CE quality):** LoRA-adapted soft-token compression retains more
  semantic content per slot than extractive or abstractive compression at the same
  compression ratio.
- **Sub-claim 3 (Reconstruction fidelity):** Anchored Progressive Reconstruction
  produces fewer hallucinations than unconstrained generative reconstruction from
  equally compressed representations.
- **Sub-claim 4 (Online learning):** End-of-session fine-tuning on missed segments
  improves L1 hit rate and downstream accuracy across sessions without catastrophic
  forgetting.

### What the paper is NOT claiming

- That this replaces RAG (different tradeoff; RAG is offline/explicit; SF is online/implicit)
- That it beats full-context inference (the upper bound; the claim is near-parity at lower cost)
- That reconstruction is lossless (L3 is a gist by design, not a copy)
- That it works at all context lengths equally (the system is a transparent passthrough below 8K tokens)

------------------------------------------------------------------------

# 1. Benchmark Categories

Three classes of tasks should be used to evaluate the system, each
testing a different property.

## Category A --- Long Context Retrieval

Purpose: Test whether important information survives compression.

### Needle-in-a-Haystack (NIAH)

The most direct test of sub-claim 1. A critical fact placed at position P in a
document of length L must be retrieved. Run across a full grid of (depth, length).

**Setup:**
- Context lengths: 4K, 8K, 16K, 32K, 64K, 128K tokens
- Needle positions: 10%, 25%, 50%, 75%, 90% into document
- Document: generic text (Paul Graham essays, Wikipedia articles)
- Needle: a synthetic specific fact ("The secret number is 42519")
- Metric: exact-match retrieval accuracy

**Target numbers:**

| System | 8K depth=10% | 32K depth=10% | 128K depth=10% |
|---|---|---|---|
| Full context | 100% | 100% | 100% |
| Truncation (last 8K) | 0% | 0% | 0% |
| StreamingLLM | ~30% | ~10% | ~5% |
| LLMLingua | ~75% | ~60% | ~40% |
| CompLLM | ~80% | ~70% | ~55% |
| **SF (target)** | **>90%** | **>85%** | **>75%** |

The key result: SF should dominate at deep positions (early needles in long documents)
because the importance scorer flags specific named facts. The NIAH heat map is
almost certainly **Figure 2** of the paper.

### NarrativeQA

QA over full books and movie scripts. Questions require content spread across the
entire document — not just the end.

- **Setup:** Full-book test split; metric: F1 score on extracted answers
- **Target:** Full context ~65 F1; SF target >62 at 25% memory; truncation ~42

### QuALITY

Multiple-choice QA on long articles. Questions are designed to require full-document
comprehension — the correct answer often depends on early content specifically.

- **Metric:** Accuracy (4-choice MC)
- **Target:** Full context ~65%, SF >60%, truncation ~45%

------------------------------------------------------------------------

## Category B --- Long-Horizon Dialogue Memory

Purpose: Evaluate whether the model remembers important conversational
facts over many turns.

### LongBench (v2 preferred)

21-task suite covering single-doc QA, multi-doc QA, summarization, few-shot
learning, and code. Provides a broad performance picture.

**Key subtasks to highlight:**
- `qasper` — scientific QA (entity-heavy, validates entity anchor region)
- `multifieldqa_en` — multi-document retrieval
- `gov_report` — summarization (validates semantic content retention)
- `narrativeqa` — as above

**Target:** SF average score within 3–5 points of full context at ≤30% memory usage.

### SCROLLS (GovReport, QMSum)

Long-document abstractive summarization. Good for measuring semantic content
retention — CE that discards too much information degrades ROUGE scores.

- **Metric:** ROUGE-1/2/L
- **Target:** Within 2 ROUGE points of full context at 4× compression

### Custom ConvRecall Benchmark (novel contribution — build this)

No existing benchmark tests multi-turn conversational memory at the granularity
this system targets. Building it is a minor standalone contribution.

**Design:**
1. Create 50 multi-turn conversation scripts (or curate from ShareGPT)
2. At turn T_intro, introduce a specific fact (a name, number, date, preference)
3. At turns T_distractor+1...T_distractor+N, insert unrelated conversation to
   trigger demotion of the T_intro segment to L2/L3
4. At T_probe, ask a question requiring the T_intro fact
5. Vary: gap length (5, 20, 50 turns), fact type (name, number, preference, instruction)

**Metric:** Recall rate at T_probe across gap lengths.
**Target:** >80% recall at 50-turn gaps. Truncation fails after ~20 turns.
**Paper framing:** Call this "SF-Bench" or "ConvRecall" — present it as a contribution
alongside the results, since the benchmark itself does not exist yet.

------------------------------------------------------------------------

## Category C --- Agent Memory and Planning

Purpose: Test interference and relevance filtering during long reasoning
processes.

Example environments:

-   ALFWorld
-   WebArena

Metrics:

-   Task success rate
-   Reasoning length
-   Context usage

This category tests the value of **active suppression**, where
irrelevant memory is inhibited. Run a system-with-suppression vs.
system-without-suppression ablation here.

------------------------------------------------------------------------

# 2. Baselines to Compare Against

All major long-context strategies must be included. Reviewers will ask about any omission.

## Baseline 1 --- Full Context (upper bound)

No compression. Provide the complete history to the LLM.

All accuracy numbers are reported as a fraction of full-context performance.
Memory usage at 1.0× is the reference point. **Run this first, record once, reuse.**

## Baseline 2 --- Recency Truncation

Keep the last N tokens. Discard everything older. This is what virtually every
production LLM chatbot does today. It is a stronger baseline than it sounds — for
many tasks, recent context is sufficient. The paper's opening argument is that this
fails precisely when early-but-critical information matters.

## Baseline 3 --- StreamingLLM

Keep attention sink tokens (first 4) + a sliding window of the last K tokens.
Enables infinite-length inference without recomputation. The closest "production
ready" system. Weakness: no importance awareness, early content lost unless it is
a sink token.

Paper: *Efficient Streaming Language Models with Attention Sinks* (Xiao et al., 2023)

## Baseline 4 --- LLMLingua / LLMLingua-2

Token-level pruning: score each token by perplexity drop, keep only the important
ones. Operates in token space (not embedding space). Lossier than soft-token
approaches but computationally cheap. Good for framing the compression-ratio tradeoff.

Paper: *LLMLingua: Compressing Prompts for Accelerated Inference of LLMs* (Jiang et al., 2023)

## Baseline 5 --- CompLLM / ICAE (most important baseline)

Soft-token compression via autoencoder — same EOS-token mechanism that SF's compressor
is based on. No hierarchy, no online learning, no importance-based allocation.

This is the most important baseline because it isolates the value of SF's additions
(hierarchy + importance scoring + online learning) over the base compression mechanism.
If SF does not beat CompLLM, the contributions of this paper are not validated.

Papers: *CompLLM*; *In-Context Autoencoder for Context Compression* (Ge et al., 2024)

## Baseline 6 --- Summarization Memory

Periodic abstractive summarization (LangChain-style): old context → LLM summary → replace.
Widely used in agents. Good contrast because it introduces a second model, has high
latency, and loses specific factual detail in the summary.

## Baseline 7 --- RAG (Dense Retrieval)

Segments stored in a vector database; top-K retrieved at query time. Fundamental
difference: offline indexing, separate retrieval model, no online learning, no
ephemeral reconstruction. Not a direct competitor but important for framing SF's
design space position.

### What to report for every baseline

| Metric | Why |
|---|---|
| Task accuracy on each benchmark | Primary result |
| Effective tokens retained | Memory cost |
| Wall-clock inference latency | Practical cost |
| Can handle context > native window | Yes/No — important for 128K+ claim |

------------------------------------------------------------------------

# 3. Key Evaluation Metrics

## Task Accuracy

Standard benchmark metrics:

- QA accuracy (exact match or F1)
- ROUGE-1/2/L for summarization
- Task success rate for agent benchmarks

## Memory Efficiency

Measure performance relative to tokens retained. The goal is a Pareto-superior curve:
higher accuracy at every memory budget compared to all baselines.

**Definition of "effective tokens retained":** L1 tokens + CE-slot equivalents from
L2 and L3 (each CE slot counts as 1 token-equivalent, since each is a single D-dim
vector injected into the LLM).

**The key figure:** Memory budget (x-axis, % of full context) vs. NarrativeQA F1
(y-axis), one Pareto curve per system. SF's curve should be above-and-left of all
others. This is **Figure 1** (teaser) for the paper.

## Cache Hit Rate

Unique metric for hierarchical memory. Track L1/L2/L3 hit rates per session and
correlate with task accuracy. A high L1 hit rate means the importance scorer
correctly predicted what would be needed. A high L3 miss-cascade rate means the
deepest compression is discarding too much.

## Reconstruction Error

Critical metric for sub-claim 3. Two components:

**Gist Retention Score (GRS):** After L3 compression (5 slots → 4× total compression),
what fraction of named entities from the original segment survive in reconstructed content?
```
GRS = |{entities in reconstruction} ∩ {entities in original}| / |{entities in original}|
```
Target: >0.80. Measures whether the entity anchor region is working.

**Miss Cascade Rate (MCR):** Fraction of L3 misses that also fail the reconstruction
gate (fingerprint similarity below θ=0.5) — total information loss events.
```
MCR = |L3 misses failing gate| / |total L3 misses|
```
Target: <10%. Above 20% means L3 compression is too aggressive for the deployment domain.

## Latency

Hierarchical memory adds overhead. Report honestly:

- Time to first token (TTFT)
- Tokens per second during generation
- L3 reconstruction latency specifically (this is the tail latency)

Compare against full context and StreamingLLM. Reviewers will ask.

------------------------------------------------------------------------

# 4. Novel Metrics Specific to This Paper

These do not exist in prior work. Introducing them is a minor contribution alongside
the system contribution — they operationalize concepts that have been informal.

### Importance-Precision@K

Of the top-K segments ranked by importance score, what fraction are actually
referenced in subsequent queries?
```
IP@K = |top-K by importance ∩ segments referenced in next 10 turns| / K
```
Compute on ConvRecall where T_intro segments are ground-truth "important." Target: >0.75.
Validates that the importance scorer is tracking what actually matters, not a proxy.

### Session Adaptation Gain (SAG)

L1 hit rate at session S minus hit rate at session 1, same domain:
```
SAG(S) = HitRate_L1(session S) − HitRate_L1(session 1)
```
Target: SAG(10) > +0.15 (15 percentage point improvement after 10 sessions).
Flat SAG means fine-tuning is not translating to better admission decisions.

------------------------------------------------------------------------

# 5. Ablations

Ablations are the most important part of a systems paper. Each one isolates exactly
one architectural decision. Without them, reviewers cannot distinguish "this works"
from "this specific component is what makes it work."

Run all ablation variants on NIAH (average across depths/lengths) and NarrativeQA F1.

| Variant | NIAH avg | NarrativeQA F1 | Notes |
|---|---|---|---|
| SF (full) | | | Baseline for ablation |
| – no online fine-tuning | | | Remove Phase 2 entirely; tests sub-claim 4 |
| – no hierarchy (flat L2 only) | | | Remove L3, compress everything uniformly; tests sub-claim 1 |
| – no importance scoring (random eviction) | | | Replace importance scorer with random; tests sub-claim 1 directly |
| – no anchors (unconstrained reconstruction) | | | Remove constraint shell from APR; tests sub-claim 3 |
| – no structured CE layout | | | Remove entity/boundary region separation; tests CE structure value |
| – C_L2=4 (aggressive compression) | | | 20 tokens → 4 CE slots (~5× compression) |
| – C_L2=12 (conservative compression) | | | 20 tokens → 12 CE slots (~1.7× compression) |
| – LoRA rank 8 | | | Smaller adapter |
| – LoRA rank 32 | | | Larger adapter |

**The most critical ablation** is "no importance scoring (random eviction)." If
random eviction matches importance-based eviction, the importance scorer is irrelevant
and the entire allocation story collapses. Expect 5–10% NIAH drop at deep positions.
If this ablation is close to full SF, investigate immediately before proceeding.

------------------------------------------------------------------------

# 6. Experimental Design

## Experiment 1 --- Compression Robustness (main table)

Dataset: LongBench + NarrativeQA + QuALITY

Compare all baselines and SF. Goal: show similar accuracy with dramatically smaller
memory usage. This is **Table 1** in the paper.

## Experiment 2 --- NIAH Heat Map

2D grid: needle depth (x-axis) × context length (y-axis). Color = retrieval accuracy.
One panel per system. SF should show high accuracy everywhere except extreme
depth × extreme length. This is **Figure 2**.

## Experiment 3 --- Online Learning Curves

X-axis: session number (1–20). Y-axis: L1 hit rate + QA accuracy on a fixed
domain-specific test set. Two lines: SF with Phase 2 fine-tuning vs. without.
The gap between lines should widen across sessions. This is **Figure 3**.

## Experiment 4 --- Reconstruction Hallucination Analysis

500 held-out Wikipedia segments compressed to L3, reconstructed, compared for
entity accuracy and factual QA accuracy. Compare: SF with APR vs. SF without
anchors vs. CompLLM reconstruction. This produces the GRS and MCR numbers. **Figure 4**.

## Experiment 5 --- Ablation Table

All ablation variants from Section 5 on NIAH + NarrativeQA. **Table 2**.

## Experiment 6 --- ConvRecall

Custom multi-turn recall benchmark. Recall rate across gap lengths (5, 20, 50 turns),
fact types. Shows the system's deployment-case strength most directly. **Figure 5** or
integrated into Table 1 as a separate section.

------------------------------------------------------------------------

# 7. Key Visualizations for the Paper

1. **Figure 1 (teaser):** Memory budget vs. NarrativeQA F1 Pareto curves. SF above-and-left.
2. **Figure 2:** NIAH heat map grid. Multiple panels per system.
3. **Figure 3:** Online learning curve — hit rate and accuracy across sessions 1–20.
4. **Figure 4:** GRS and entity substitution rate by system. Bar chart.
5. **Figure 5:** Ablation summary. Grouped bars for NIAH + NarrativeQA F1 per variant.

------------------------------------------------------------------------

# 8. Anticipated Reviewer Objections

Prepare experiments to pre-empt these before submission.

**"Just use a larger context window."**
Show results at 128K and 256K tokens where full attention is prohibitively expensive.
Compute scales quadratically with sequence length; SF scales near-linearly (only L1
pays full attention cost).

**"How does this compare to RAG?"**
Include RAG in Table 1. Note qualitative difference: RAG requires offline indexing, a
separate retrieval model, and cannot compress streaming conversation in-place.

**"Is the improvement from the hierarchy or the learned compression?"**
The "no hierarchy" ablation isolates this. If flat L2-only compression matches full
SF within 2%, the hierarchy is not load-bearing — re-center the claim on CE quality.

**"Online learning could catastrophically forget."**
Run Phase 1 held-out L_distill evaluation after every 10 sessions in the multi-session
experiment. If L_distill on Phase 1 test data does not rise, forgetting is not occurring.
Report this explicitly as a "forgetting analysis" paragraph — don't leave reviewers to
wonder.

**"APR hallucination mitigation is a heuristic, not a guarantee."**
Honest framing: yes, APR reduces hallucination risk; it does not eliminate it. Report
MCR explicitly. Do not overclaim.

------------------------------------------------------------------------

# 9. Target Numbers Summary

| Benchmark | Full context | Best prior (CompLLM est.) | **SF target** | Memory budget |
|---|---|---|---|---|
| NIAH avg (all depths/lengths) | 100% | ~70% | **>85%** | ~25% of full context |
| NarrativeQA F1 | ~65 | ~60 | **>62** | ~25% |
| LongBench avg (normalized) | 100% | ~90% | **>93%** | ~25% |
| QuALITY accuracy | ~65% | ~58% | **>62%** | ~25% |
| GRS (entity retention at L3) | 100% | ~65% | **>80%** | L3 segments only |
| MCR (miss cascade rate) | N/A | N/A | **<10%** | — |
| IP@K (importance precision) | N/A | N/A | **>0.75** | — |
| SAG(10) (session adaptation) | N/A | N/A | **>+0.15** | — |

------------------------------------------------------------------------

# 10. Core Claim to Validate

The experiments should ultimately support the claim:

> Not all tokens deserve equal memory priority — and a system that has learned
> which tokens matter, and can refine that judgment from its own deployment
> failures, achieves near-full-context performance at a fraction of the memory cost.

If the system demonstrates **importance-aware memory allocation improves efficiency
without degrading accuracy**, and **improves further with deployment**, the approach
becomes a compelling research contribution with no prior art on the second half.