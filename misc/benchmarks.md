# Benchmarks and Baselines for Evaluating Semantic Forgetfulness

## Goal of the Experiments

This project should be evaluated as a **memory systems paper**, not
purely as a language model paper.

The central claim to test:

> Hierarchical semantic memory improves long-horizon reasoning
> efficiency while maintaining task performance.

Experiments should demonstrate that selective memory allocation
outperforms existing long-context strategies when context length grows
very large.

------------------------------------------------------------------------

# 1. Benchmark Categories

Three classes of tasks should be used to evaluate the system, each
testing a different property.

## Category A --- Long Context Retrieval

Purpose: Test whether important information survives compression.

Typical tasks include:

-   Needle-in-a-Haystack retrieval
-   Long-document question answering
-   Multi-document reasoning

Representative benchmarks:

-   Needle-in-a-Haystack
-   NarrativeQA
-   QuALITY

Metrics:

-   Recall of early information
-   Question answering accuracy
-   Robustness to long context

Failure on these tasks would indicate the system cannot reliably
preserve important information.

------------------------------------------------------------------------

## Category B --- Long-Horizon Dialogue Memory

Purpose: Evaluate whether the model remembers important conversational
facts over many turns.

Relevant benchmarks:

-   LongBench
-   SCROLLS

Tasks include:

-   Dialogue memory
-   Long-context summarization
-   Long-document reasoning

Metrics:

-   Factual recall across turns
-   Dialogue coherence
-   Memory efficiency (tokens required)

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
irrelevant memory is inhibited.

------------------------------------------------------------------------

# 2. Baselines to Compare Against

All major long-context strategies must be included.

## Baseline 1 --- Raw Long Context

No compression.

Simply provide the entire context window to the model.

Purpose:

Determine whether compression is necessary.

------------------------------------------------------------------------

## Baseline 2 --- Summarization Memory

A common approach used in many LLM agents.

Strategy:

conversation → periodic summaries → replace older messages

Used in systems such as:

-   LangChain
-   AutoGPT

------------------------------------------------------------------------

## Baseline 3 --- Retrieval Augmented Generation (RAG)

Documents stored externally and retrieved when needed.

Evaluation metrics:

-   Retrieval accuracy
-   Downstream QA performance

------------------------------------------------------------------------

## Baseline 4 --- Prompt / Memory Compression

Research systems designed to reduce context size.

Examples include:

-   CompLLM
-   FastKV

These provide the closest comparison to the proposed architecture.

------------------------------------------------------------------------

# 3. Key Evaluation Metrics

## Task Accuracy

Standard benchmark metrics:

-   QA accuracy
-   F1 score
-   Task success rate

------------------------------------------------------------------------

## Memory Efficiency

Measure performance relative to tokens retained.

Example metric:

Accuracy vs Tokens Used

The goal is to demonstrate higher accuracy with significantly fewer
tokens.

------------------------------------------------------------------------

## Cache Hit Rate

Unique metric for hierarchical memory.

Track:

-   L1 hit rate
-   L2 hit rate
-   L3 hit rate

Correlate hit rates with task performance.

------------------------------------------------------------------------

## Reconstruction Error

Critical safety metric.

Measure accuracy of reconstructed content from compressed memory.

Example:

Reconstruction F1 vs Compression Ratio

------------------------------------------------------------------------

## Latency

Hierarchical memory introduces overhead.

Measure:

-   Query latency
-   Tokens per second
-   End-to-end inference time

------------------------------------------------------------------------

# 4. Experimental Design

## Experiment 1 --- Compression Robustness

Dataset:

LongBench QA tasks

Compare:

-   Raw context
-   Summarization memory
-   RAG
-   Compression baselines
-   Semantic Forgetfulness

Goal:

Show similar accuracy with dramatically smaller memory usage.

------------------------------------------------------------------------

## Experiment 2 --- Long Dialogue Memory

Simulate conversations lasting hundreds or thousands of turns.

Evaluate:

-   Recall of early conversation facts
-   User preference consistency
-   Dialogue coherence

------------------------------------------------------------------------

## Experiment 3 --- Agent Task Interference

Use an autonomous agent benchmark.

Compare:

-   System with active suppression
-   System without suppression

Goal:

Demonstrate that suppression reduces irrelevant memory retrieval.

------------------------------------------------------------------------

# 5. Key Visualization for the Paper

The most important plot should be:

Task Accuracy vs Memory Size

A convincing result would show:

-   Similar or better accuracy
-   Significantly smaller memory footprint

------------------------------------------------------------------------

# 6. Potential Custom Benchmark

Existing benchmarks do not fully evaluate importance-aware forgetting.

A custom benchmark could include:

-   Very long documents (100k+ tokens)
-   A small number of critical facts embedded in noise
-   Questions referencing those facts

Goal:

Evaluate whether models preserve high-importance information during
compression.

------------------------------------------------------------------------

# 7. Core Claim to Validate

The experiments should ultimately support the claim:

> Not all tokens deserve equal memory priority.

If the system demonstrates that **importance-aware memory allocation
improves efficiency without degrading accuracy**, the approach becomes a
compelling research contribution.
