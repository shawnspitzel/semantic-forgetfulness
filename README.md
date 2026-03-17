# Semantic Forgetfulness
Current LLMs treat the context window as a flat buffer. When it fills, old content is truncated regardless of importance. As context grows, attention dilutes and early-but-critical information gets buried. We call this *context rot*.

Existing approaches (KV pruning, sliding windows, static compression) reduce memory footprint, but none of them *learn* what a given session needs. They apply fixed policies to a dynamic problem.

## The Approach

Semantic Forgetfulness implements a three-tier memory hierarchy — L1 (active context, GPU), L2 (compressed working set, CPU), L3 (aggressively compressed gist store, CPU), where content is allocated based on semantic importance rather than recency.

The main insight here is that a cache miss is a training signal. If the model demoted a segment and later needed it, that was a bad allocation decision. Those mistakes fine-tune the importance model, making the system progressively better at knowing what to keep. This continuous learning model is used during deployment, so this adaptation happens on the fly.

Content is compressed into dense *Concept Embeddings* (CEs, inspired by CompLLM) that live in the model's own token embedding space and are directly injectable as soft tokens. A companion reconstructor uses *Anchored Progressive Reconstruction (APR)* to recover content from gists, with hard constraints (boundary sentences, named entities, semantic fingerprints) to prevent hallucination.
