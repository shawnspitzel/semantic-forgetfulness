# Reconstructor Module

> The reconstructor's sole responsibility is recovering semantic meaning from a compressed gist (CE tensor) during cache promotion — L3→L2 and L2→L1. It operates entirely in embedding space. It does not decode to text at any stage. It does not decide when to run; that is the cache controller's concern. It does not freely generate; every output is bounded by hard anchors derived from the segment's sanity anchors. Its goal is fidelity, not richness.

---

## Design Philosophy

Reconstruction is a **narrowing funnel**, not a generation task. The reconstructor is given a compressed CE tensor and asked to recover what was lost — not to invent what might have been there. Every degree of freedom it has is an opportunity to hallucinate. The architecture is designed to minimize degrees of freedom at every stage.

Three principles govern all design decisions:

1. **Hard constraints before soft generation** — what we know with certainty is locked in first
2. **Ground in real context, not model priors** — decompression is anchored to actual L2 neighbor CE tensors, not the model's parametric knowledge
3. **Fidelity over richness** — a sparse but accurate reconstruction is strictly better than an elaborated but hallucinated one

---

## Anchored Progressive Reconstruction (APR)

APR is the reconstruction framework. It has three layers that operate in sequence on every reconstruction call. All operations are in embedding space — no text is produced at any stage.

### Layer 1: Constraint Shell

Applied unconditionally at every stage, L3→L2 and L2→L1.

- The **boundary region** (CE[E:E+B]) is directly initialized from the tokenized embeddings of `sanity_anchors.boundary_sentences`. This region is not generated — it is set. The reconstructor does not touch it beyond initialization.
- The **entity anchor region** (CE[0:E]) must satisfy a cosine similarity check: each entity slot must have cosine similarity ≥ τ to its corresponding stored entity embedding (from `sanity_anchors.entities`). Any candidate output where an entity slot fails this check is rejected and retried (up to a fixed retry budget). On exhaustion, the entity region is initialized directly from the stored entity embeddings — the embedding-space analog of falling back to boundary sentences + entity list only.
- The `semantic_fingerprint` cosine check is the **exit gate**: output only passes if `cosine(output_representative_vec, semantic_fingerprint) ≥ θ`. Outputs that fail this check are not promoted.

The constraint shell bounds the failure mode. The reconstructor cannot produce a CE that is semantically unrecognizable, cannot omit entity anchors, and cannot misframe the boundary region. Hallucination is bounded to the semantic content region.

### Layer 2: Structured CE Format

The CE tensor has a fixed internal layout that the reconstructor reads and writes deliberately:

```
CE[0 : E]       → entity anchors region    (E slots, one per key entity at fixed positions)
CE[E : E+B]     → boundary region          (encodes first/last sentence semantics)
CE[E+B : D]     → semantic content region  (free latent space for gist)
```

- **Entity region** — constrained by Layer 1. Initialized from stored entity embeddings; refined by the decompressor within the cosine similarity constraint.
- **Boundary region** — set directly from boundary sentence token embeddings. Not refined.
- **Semantic content region** — the only region with degrees of freedom. Filled by interpolating between the source CE's semantic content region and the grounding signal from L2 neighbors.

**Training objective**: compressor and reconstructor are trained jointly as an autoencoder pair.

```
Original segment tokens
      ↓
Compressor → CE tensor (structured format)
      ↓
Reconstructor → Decompressed CE tensor
      ↓
Loss: reconstruction fidelity (cosine similarity between decompressed CE and original token embeddings)
    + injection quality loss (CE remains valid as soft tokens in LLM forward pass)
```

The compressor learns to fill the structured regions in ways the reconstructor can decompress. The injection quality loss ensures the CE remains valid for direct prepending to the LLM forward pass. Neither objective is sacrificed.

### Layer 3: Context-Grounded Expansion

When L2 is warm, the reconstructor queries the entity graph for CE tensors in L2 that are neighbors of the segment being reconstructed. These neighbor CEs serve as **grounding context** — real compressed representations the reconstructor interpolates toward when filling the semantic content region.

The reconstructor is not expanding the CE from the source CE alone; it is interpolating between the source CE's semantic content region and the semantic content regions of verified L2 neighbors. This is the primary mechanism for suppressing confabulation.

If no L2 neighbors are available (e.g., during the early warm-start window), the reconstructor fills the semantic content region from the source CE alone — a sparser but still constrained output.

---

## Two-Stage Pipeline

### Stage 1: L3 → L2 (Fidelity Pass)

Goal: produce a mid-compression CE with high entity and boundary fidelity. Do not elaborate beyond what is anchored.

```
Input:  L3 CE tensor (high compression), sanity_anchors, L2 neighbor CEs (if available)
Output: L2 CE tensor (mid-compression)

Process:
  1. Initialize boundary region from stored boundary sentence token embeddings (locked)
  2. Initialize entity anchor region from stored entity embeddings
  3. Query entity graph → retrieve L2 neighbor CEs as grounding context
  4. Fill semantic content region: interpolate between source CE semantic region and neighbor CE semantic regions
     — prefer neighbor grounding when available
     — fall back to source CE alone if no neighbors found
  5. Entity anchor check → cosine sim ≥ τ for each entity slot; retry if any fail (max 3 retries)
     — on exhaustion: initialize entity region directly from stored entity embeddings
  6. Semantic fingerprint check → reject if cosine(output_representative_vec, semantic_fingerprint) < θ
  7. Emit L2 CE tensor with per-slot confidence scores
```

The L3→L2 output is intentionally constrained. It recovers what it can verify from anchors and neighbors.

### Stage 2: L2 → L1 (Query-Conditioned Injection)

Goal: refine the L2 CE toward the current query's demands, then inject directly into L1 as soft tokens.

```
Input:  L2 CE tensor, current query vector, L2 neighbor CEs
Output: CE tensor injected directly into L1 token buffer as soft tokens

Process:
  1. Re-apply constraint shell (boundary region locked; entity anchor cosine check)
  2. Compute query-relevance score for each CE slot in the semantic content region
  3. For high query-relevance slots: refine using L2 neighbor semantic regions as grounding
  4. For low query-relevance slots: leave as-is from the L2 CE
  5. Semantic fingerprint check → reject if cosine < θ
  6. Prepend output CE directly to LLM forward pass as soft tokens — no decode to text
```

The query-conditioning is the key property of this stage. The same L3 segment may be refined differently depending on what the user actually needed. This mirrors context-dependent recall: the same memory yields different detail depending on the retrieval cue.

---

## Confidence Scores

The reconstructor emits a **per-slot confidence score** alongside every CE output. These scores are observability instrumentation — they do not gate promotion (the semantic fingerprint check does that).

Confidence is measured as the cosine similarity between each output CE slot and the expected anchor value (entity embedding for entity slots; boundary embedding for boundary slots; neighbor CE interpolation target for semantic slots).

Confidence scores are used for:
- Debug traces: log which CE slots the reconstructor was uncertain about on any given promotion
- Training signal: low-confidence slots that turn out accurate indicate over-caution; high-confidence slots that turn out wrong indicate hallucination hotspots in the semantic content region
- Offline analysis: aggregate confidence distributions across sessions to identify systematic failure modes in the compressor

Confidence scores are stored in the `L2Entry` metadata at promotion time and discarded at L1 promotion.

---

## Warm-Start Protocol

> The warm-start protocol is enforced by the **cache controller**, not the reconstructor. It is documented here because it is the prerequisite for Layer 3 (context-grounded expansion) to function correctly.

**The cold-start problem**: if L2 is initially empty or sparse, the entity graph returns no neighbors and the reconstructor fills the semantic content region from the source CE alone — which is acceptable but lower fidelity.

**The solution**: L3 demotion is gated behind a minimum L2 population threshold. For the first K segments of any session, demotion to L3 is disabled. Segments go directly into L2 as mid-compression CEs.

**Why this works**: L2 is seeded from L1 demotions — the active context, which is always high quality. By the time L3 demotion begins, L2 contains a reliable grounding corpus of CE tensors from real, attended-over content.

---

## Interfaces

### Input to Reconstructor

```
reconstruct(
  ce_tensor:            Tensor[C, D],        # structured CE from L3Entry or L2Entry
  sanity_anchors:       SanityAnchors,       # boundary_sentences (for token embeddings), entities (for entity embeddings), semantic_fingerprint
  l2_neighbors:         List[L2Entry],       # entity graph neighbors currently in L2 (may be empty)
  query_vec:            Tensor[D] | None,    # present only for L2→L1 stage
  stage:                "l3_to_l2" | "l2_to_l1"
) → ReconstructionResult
```

### Output from Reconstructor

```
ReconstructionResult {
  ce_tensor:            Tensor[C2, D]        # decompressed CE (C2 ≥ C; more slots = less compression)
  confidence_scores:    List[(slot, float)]  # per-slot cosine similarity to anchor target
  fingerprint_sim:      float                # cosine similarity against semantic_fingerprint
  grounding_used:       bool                 # whether L2 neighbor CEs were available and used
  fallback:             bool                 # true if entity region initialized directly from stored embeddings
}
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| No text decode | All operations in embedding space | Text is a lossy intermediate when the endpoint is the token context window; CEs inject directly |
| Boundary region | Directly initialized from boundary sentence token embeddings | Eliminates hallucination at the frame level without requiring text generation |
| Entity enforcement | Cosine similarity check per entity slot, retry on failure | Preserves factual anchoring in embedding space; same retry logic as prior text-based enforcement |
| CE format | Structured regions (entity / boundary / semantic) | Allows reconstructor to operate deliberately on each region rather than treating the CE as a flat vector |
| Training objective | Joint autoencoder + injection quality | Preserves both decodability and LLM-injectability |
| L3→L2 pass | Fidelity-first, constrained decompression | Builds a trustworthy L2 CE before query-conditioned refinement |
| L2→L1 pass | Query-conditioned refinement + direct soft token injection | Generates only what the retrieval cue demands; no text step |
| Grounding source | L2 neighbor CE semantic regions via entity graph | Grounds decompression in real verified content, not model priors |
| Confidence scores | Per-slot cosine similarity to anchor target | Observability without gating; fingerprint check is the gate |
| Warm-start ownership | Cache controller, not reconstructor | Reconstructor assumes L2 is warm; controller guarantees it |
