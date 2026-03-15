# Reconstructor Module

> The reconstructor's sole responsibility is recovering semantic meaning from a compressed gist (CE tensor) during cache promotion — L3→L2 and L2→L1. It operates as a constrained, context-grounded module. It does not decide when to run; that is the cache controller's concern. It does not freely generate; every output is bounded by hard anchors. Its goal is fidelity, not richness.

---

## Design Philosophy

Reconstruction is a **narrowing funnel**, not a generation task. The reconstructor is given a compressed representation and asked to recover what was lost — not to invent what might have been there. Every degree of freedom it has is an opportunity to hallucinate. The architecture is designed to minimize degrees of freedom at every stage.

Three principles govern all design decisions:

1. **Hard constraints before soft generation** — what we know with certainty is locked in first
2. **Ground in real context, not model priors** — generation is anchored to actual L2 content, not the model's parametric knowledge
3. **Fidelity over richness** — a sparse but accurate reconstruction is strictly better than an elaborated but hallucinated one

---

## Anchored Progressive Reconstruction (APR)

APR is the reconstruction framework. It has three layers that operate in sequence on every reconstruction call.

### Layer 1: Constraint Shell

Applied unconditionally at every stage, L3→L2 and L2→L1.

- `boundary_sentences` are copied **verbatim** into the output. The reconstructor does not touch them.
- `entities` from `sanity_anchors` are enforced as a **hard inclusion list**. Any candidate output that does not contain all listed entities is rejected and retried (up to a fixed retry budget; on exhaustion, fall back to boundary sentences only).
- The `semantic_fingerprint` cosine check is the **exit gate**: output only passes if `cosine(output_embedding, semantic_fingerprint) ≥ θ`. Outputs that fail this check are not promoted.

The constraint shell bounds the failure mode. The reconstructor cannot invent new named entities, cannot contradict the framing sentences, and cannot produce a semantically unrecognizable output. Hallucination is bounded to the interior fill.

### Layer 2: Structured CE Format

The CE tensor is redesigned for decodability. Rather than a flat dense vector optimized purely for LLM injection, the CE has a **structured internal layout**:

```
CE[0 : E]       → entity anchors region    (E slots, one per key entity at fixed positions)
CE[E : E+B]     → boundary region          (encodes first/last sentence semantics)
CE[E+B : D]     → semantic content region  (free latent space for gist)
```

The reconstructor knows the layout and reads each region deliberately:
- Entity region → confirms what entities must appear
- Boundary region → frames the reconstruction
- Semantic content region → informs interior fill

**Training objective**: compressor and reconstructor are trained jointly as an autoencoder pair.

```
Original segment
      ↓
Compressor → CE tensor (structured format)
      ↓
Reconstructor → Reconstructed segment
      ↓
Loss: reconstruction fidelity on held-out segments
    + injection quality loss (CE still usable as soft tokens in LLM forward pass)
```

The compressor learns to fill the structured regions in ways the reconstructor can decode. The injection quality loss ensures the CE remains valid for its original use case — direct prepending to the LLM forward pass. Neither objective is sacrificed.

### Layer 3: Context-Grounded Expansion

When L2 is warm (see Warm-Start Protocol below), the reconstructor queries the entity graph for segments in L2 that are neighbors of the segment being reconstructed. These neighbors serve as **grounding context** — real text the reconstructor can draw from.

The grounding context is used to fill the interior of the reconstruction, between the locked boundary sentences. Crucially, the reconstructor is not generating from the CE alone; it is interpolating between the CE signal and real verified content from L2. This is the primary mechanism for suppressing confabulation.

If no L2 neighbors are available (e.g., during the early warm-start window), the reconstructor falls back to boundary sentences + entity list only — a minimal but trustworthy skeleton.

---

## Two-Stage Pipeline

### Stage 1: L3 → L2 (Fidelity Pass)

Goal: produce a sparse, high-confidence skeleton. Do not elaborate beyond what is anchored.

```
Input:  CE tensor, sanity_anchors, L2 neighbor segments (if available)
Output: skeleton reconstruction (boundary sentences + entity-filled interior)

Process:
  1. Lock boundary_sentences into output frame
  2. Read CE entity region → verify against entities list
  3. Query entity graph → retrieve L2 neighbors as grounding context
  4. Fill interior: interpolate between CE semantic region and grounding context
     — prefer grounding context over CE signal when both are available
     — fall back to CE signal alone if no neighbors found
  5. Enforce entity inclusion → retry if any entity missing (max 3 retries)
  6. Semantic fingerprint check → reject if cosine < θ
  7. Emit reconstruction with per-span confidence scores
```

The L3→L2 output is intentionally lean. It does not attempt to recover everything — it recovers what it can verify.

### Stage 2: L2 → L1 (Query-Conditioned Enrichment Pass)

Goal: enrich the skeleton with content relevant to the current query. Generate only what the query demands.

```
Input:  L2 skeleton reconstruction, current query vector, L2 neighbor segments, CE tensor
Output: enriched reconstruction targeted at the query

Process:
  1. Re-apply constraint shell (boundary sentences + entity enforcement)
  2. Compute query-relevance score against each span of the L2 skeleton
  3. For spans with high query-relevance: expand using L2 neighbors as grounding
  4. For spans with low query-relevance: leave as-is from the L2 skeleton
  5. Semantic fingerprint check → reject if cosine < θ
  6. Emit reconstruction with updated per-span confidence scores
```

The query-conditioning is the key property of this stage. The same L3 segment may be enriched differently depending on *why* it was surfaced — what the user actually needed. This mirrors context-dependent recall: the same memory yields different detail depending on the retrieval cue.

---

## Confidence Scores

The reconstructor emits a **per-span confidence score** alongside every reconstruction. These scores are observability instrumentation — they do not gate promotion (the semantic fingerprint check does that).

Confidence scores are used for:
- Debug traces: log which spans the reconstructor was uncertain about on any given promotion
- Training signal: low-confidence spans that turn out accurate indicate over-caution; high-confidence spans that turn out wrong indicate hallucination hotspots
- Offline analysis: aggregate confidence distributions across sessions to identify systematic failure modes in the compressor

Confidence scores are stored in the `L2Entry` metadata at promotion time and discarded at L1 promotion (L1 is assumed to contain verified content).

---

## Warm-Start Protocol

> The warm-start protocol is enforced by the **cache controller**, not the reconstructor. It is documented here because it is the prerequisite for Layer 3 (context-grounded expansion) to function correctly.

**The cold-start problem**: if L2 is initially empty or populated with low-quality segments, retrieval-augmented reconstruction from L2 actively grounds reconstructions in bad data — worse than grounding in nothing.

**The solution**: L3 demotion is gated behind a minimum L2 population threshold.

- For the first K segments of any session, demotion to L3 is disabled. Segments go directly into L2.
- Once L2 reaches the minimum population threshold, normal demotion policy resumes.
- K is governed by the `conservity` hyperparameter.

**Why this works**: L2 is initially populated from L1 demotions — the active context, which is always high quality. The warm start is never seeded with random or garbage content; it is seeded with what the model was actually attending to. By the time L3 demotion begins, L2 contains a reliable grounding corpus.

This mirrors how human conversational memory bootstraps: early utterances are treated as maximally important by default (nothing to compare against), and importance is re-ranked dynamically as more context accumulates.

---

## Interfaces

### Input to Reconstructor

```
reconstruct(
  ce_tensor:            Tensor[D],           # structured CE from L3Entry
  sanity_anchors:       SanityAnchors,       # boundary_sentences, entities, semantic_fingerprint
  l2_neighbors:         List[L2Entry],       # entity graph neighbors currently in L2 (may be empty)
  query_vec:            Tensor[D] | None,    # present only for L2→L1 stage
  stage:                "l3_to_l2" | "l2_to_l1"
) → ReconstructionResult
```

### Output from Reconstructor

```
ReconstructionResult {
  text:                 str                  # reconstructed segment text
  confidence_scores:    List[(span, float)]  # per-span confidence
  fingerprint_sim:      float                # cosine similarity against semantic_fingerprint
  grounding_used:       bool                 # whether L2 neighbors were available and used
  fallback:             bool                 # true if fell back to boundary sentences only
}
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Boundary sentences | Copied verbatim, never generated | Eliminates hallucination at the frame level |
| Entity enforcement | Hard inclusion list, retry on failure | Entities are the primary factual anchors |
| CE format | Structured regions (entity / boundary / semantic) | Allows reconstructor to read intentionally rather than decode a flat vector |
| Training objective | Joint autoencoder + injection quality | Preserves both decodability and LLM-injectability |
| L3→L2 pass | Fidelity-first, minimal elaboration | Builds a trustworthy base for L2→L1 enrichment |
| L2→L1 pass | Query-conditioned enrichment | Generates only what the retrieval cue demands |
| Grounding source | L2 neighbors via entity graph | Grounds generation in real verified content, not model priors |
| Confidence scores | Observability only, not promotion logic | Semantic fingerprint check is the gate; confidence informs debugging and training |
| Warm-start ownership | Cache controller, not reconstructor | Separation of concerns: reconstructor assumes L2 is warm, controller guarantees it |
