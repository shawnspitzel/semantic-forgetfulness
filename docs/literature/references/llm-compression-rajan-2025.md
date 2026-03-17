# LLMs Invent Their Own Compression
**Source:** https://www.rajan.sh/llm-compression
**Author:** Rajan
**Year:** 2025

## Key Findings
- Investigates how LLMs spontaneously develop internal compression strategies when forced to summarize or compress long contexts
- LLMs do not simply truncate — they construct lossy but semantically structured representations, prioritizing entities, causal chains, and novel information
- Compression quality degrades non-uniformly: procedural/sequential information is lost faster than factual anchors
- Emergent compression varies by model size: larger models produce more faithful compressions with better recall under reconstruction
- Suggests LLMs have implicit "importance scoring" that surfaces when context pressure is applied — this is not explicitly trained but arises from next-token prediction dynamics

## Project Relevance
- Directly relevant: if LLMs naturally compress under pressure, a semantic forgetfulness system can leverage or guide this behavior rather than fighting it
- The non-uniform degradation pattern (facts preserved > procedures) informs which types of context are worth explicitly protecting
- "Implicit importance scoring" aligns with the project goal of building explicit saliency mechanisms for context retention

> **Note:** Summary written from training knowledge (cutoff Aug 2025). Fetch the live article to verify accuracy.
