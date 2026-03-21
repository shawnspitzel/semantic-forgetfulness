# Semantic Forgetfulness — Pretraining Guide

A deep guide for training the compressor and reconstructor to produce high-quality
Concept Embeddings (CEs). This document covers dataset selection, the full metrics
landscape, key signals to watch, and observability infrastructure.

For the quick operational walkthrough (install, run, checkpoint), see
[training-guide.md](training-guide.md).

---

## 1. What "pretraining" means here

The compressor and reconstructor share the same frozen Llama backbone. Pretraining
teaches their LoRA adapters + linear projections to do two things simultaneously:

1. **CE-space alignment** — the CE slots the compressor produces must live in the
   same latent subspace as the LLM's token embeddings, so the frozen backbone can
   attend to them without distribution shift. This is the job of **L_distill**.

2. **Semantic preservation** — the CE slots must retain enough semantic content for
   the reconstructor to recover a credible approximation of the original. This is
   the job of **L_recon**.

These objectives are complementary but not identical. A CE can align well in latent
space while still discarding semantically important structure, or vice versa. Good
pretraining requires both to converge. This is why dataset choice matters: the data
must provide signal for both objectives across the CE's three structured regions
(entity, boundary, semantic content).

---

## 2. Recommended datasets

### 2.1 What properties the data needs

| Property | Why it matters |
|---|---|
| **Long documents (1,000+ words each)** | Every document is chunked into ~20-token segments. Short documents give too few segments per forward pass to build a meaningful distillation signal. Aim for documents with at least 50 segments (≈1,000 tokens). |
| **Entity-rich text** | The CE layout dedicates 2 slots (`E=2`) to named entity anchors. Training data must contain named people, places, organizations, and dates in sufficient density to teach the entity region what to encode. |
| **Varied sentence boundaries** | The boundary region (2 slots, `B=2`) is initialized from the first and last sentence of each segment. Varied boundary styles (questions, declaratives, incomplete sentences) make the boundary region robust. |
| **Conversational / instructional text** | The system's deployment context is multi-turn conversation. Domain mismatch between pretraining and deployment degrades L_distill, since the LLM's hidden states have different structure for narrative prose vs. dialogue. Include at least 30% conversational data. |
| **Multi-domain** | Generalization across domains (code, legal, medical, fiction, chat) prevents the CE space from collapsing to a narrow manifold. |

### 2.2 Tier-1 datasets (high-priority)

These cover the critical properties with minimal preprocessing.

**OpenWebText2 (The Pile)**
- ~65GB of English web text, high-quality filtered
- Entity density: moderate
- Document length: excellent (avg ~800 words)
- Domain: news, Wikipedia, blogs
- Ideal for: L_distill baseline, semantic content region
- Source: `EleutherAI/pile` on Hugging Face, `pile_subset=openwebtext2`

**Wikipedia (20220301.en)**
- Encyclopedic text with very high entity density
- Every article contains named entities, dates, organizations
- Document length: varies (use only articles >500 words)
- Ideal for: entity slot training, boundary sentence variety
- Source: `wikimedia/wikipedia` on Hugging Face

**ShareGPT / UltraChat-200k**
- Human-assistant conversation pairs, multi-turn
- Matches deployment distribution most closely
- Entity density: low-medium (personal names, products)
- Document length: short — concatenate full conversation threads
- Ideal for: L_inject signal, conversational CE alignment
- Sources: `anon8231489123/ShareGPT_Vicuna_unfiltered`, `HuggingFaceH4/ultrachat_200k`

**Project Gutenberg (Books)**
- Long narrative documents (10,000–200,000 words each)
- Excellent for training across very long context ranges
- Ideal for: L2→L3 compression (the 4x compression ratio)
- Source: `pg19` on Hugging Face (standard NLP benchmark subset)

### 2.3 Tier-2 datasets (recommended additions)

**SCROLLS benchmark corpora**
- GovReport: ~1,500 word government reports, structured and entity-dense
- SummScreenFD: TV show transcripts — high conversational density
- QMSum: meeting transcripts, good for cross-turn reference training
- Source: `tau/scrolls` on Hugging Face

**NarrativeQA**
- QA pairs over long books and movie scripts
- Provides natural supervision for L_inject: the LLM must answer questions
  correctly when given CE-compressed context
- The QA label is the cleanest available L_inject signal
- Source: `deepmind/narrativeqa` on Hugging Face

**OpenAssistant (OASST2)**
- Human feedback annotation on conversations
- Provides quality signal on conversational CE alignment
- Source: `OpenAssistant/oasst2` on Hugging Face

### 2.4 Tier-3 datasets (domain specialization)

Add these only if the deployment target is domain-specific:

| Domain | Dataset |
|---|---|
| Code | `codeparrot/github-code` (filter to README + docstrings — prose only) |
| Legal | `pile-of-law/pile-of-law` |
| Medical | `pubmed_abstracts` or `medqa-usmle` |
| Scientific | `allenai/s2orc` (abstract + intro sections) |

### 2.5 Dataset mixing strategy

For a Phase 1 training run targeting Llama-3.2-3B on conversational deployment:

```
40%  OpenWebText2         # L_distill baseline, semantic content
25%  Wikipedia            # Entity slot training
20%  ShareGPT/UltraChat   # Deployment distribution match
10%  PG-19 books          # Long-context compression stress test
 5%  NarrativeQA          # L_inject supervision (QA-labeled)
```

Build this as a `data/train.txt` concatenation, or modify `pretrain.py` to use a
`datasets` interleaved loader (see Section 7 for the implementation sketch).

### 2.6 Target corpus size

| Phase | Minimum | Recommended | Notes |
|---|---|---|---|
| Phase 1 (pretraining) | 50K tokens (fallback) | 10M–100M tokens | Larger = better generalization; diminishing returns after ~50M |
| Phase 2 (fine-tuning) | 20 miss events | 50–300 miss events | More = more gradient signal; single step regardless |

The 500-step default in `pretrain.py` covers ~10,000 segments with a 20-token target
length — about 200,000 tokens. This is sufficient to verify the pipeline, but not
enough for meaningful convergence. For serious pretraining, target 50,000–500,000 steps.

---

## 3. Metrics to optimize

### 3.1 The three training losses

#### L_distill — Hidden activation alignment

```
L_distill = mean over mid-to-late layers of:
  SmoothL1( h_ce / std(h_full),  h_full / std(h_full) )
```

**What it measures:** How closely the LLM's hidden states at layers `[num_layers/2 : num_layers]`
match when attending to CE slots vs. attending to the full original token sequence.

**Target trajectory:**
- Start: 0.5–1.5 (random LoRA initialization)
- After 1K steps: 0.3–0.5
- Converged (50K+ steps): < 0.15
- Excellent: < 0.05

**Interpretation:**
- Values > 1.0 after 5K steps → learning rate too high, or CE slots are
  too few to cover the segment's information
- Plateau above 0.3 after 10K steps → try increasing `C_L2`, reducing segment
  length, or increasing LoRA rank
- Oscillation without descent → AdamW weight decay too high, or gradient clipping needed

**Why it's the primary signal:** L_distill failure means CEs are not valid soft tokens
for the backbone. L_recon failure is recoverable; L_distill failure breaks the entire
injection mechanism.

#### L_recon — Reconstruction fidelity

```
L_recon = 1 - cosine_similarity( mean(reconstructor(CE)), mean(original_token_embeddings) )
```

**What it measures:** How well the reconstructor's output, averaged over CE slots,
points in the same direction as the original token embeddings.

**Target trajectory:**
- Start: 0.3–0.5 (random init; cosine distance from a random direction)
- After 1K steps: 0.2–0.3
- Converged: < 0.1
- Excellent: < 0.05

**Interpretation:**
- Values > 0.4 after convergence → entity/boundary regions are consuming too much
  capacity; reduce `E` or `B`; or the reconstruction is collapsing to a shared mean
- Near-zero early (< 1K steps) → check for mean-collapse: both means pointing
  at the same token embedding centroid (common with short fallback data)

**Note on mean-based metric:** This metric measures mean alignment only, not
per-slot structure. It is necessary but not sufficient. A CE that encodes the
correct mean but shuffles structure across slots will pass L_recon but fail at
query-conditioned reconstruction. See Section 3.3 for per-slot metrics.

#### L_inject — Task performance preservation (deferred)

```
L_inject = task_loss(LLM with CE injected) - task_loss(LLM with original tokens)
```

**What it measures:** Whether CE injection degrades downstream task performance.

**Current status:** L_inject is not implemented in `pretrain.py`. The NarrativeQA
subset (recommended above) is the natural place to add it: use QA pairs as supervision
and compare CE-context vs. original-context answer likelihoods.

**Target:** Δ < 0.05 perplexity increase at convergence. Values above 0.2 indicate
the CE is semantically viable (passes L_distill + L_recon) but loses task-critical
structure.

**When to add it:** After L_distill and L_recon have converged to their targets.
Adding L_inject too early introduces a conflicting gradient that destabilizes early
training.

### 3.2 Operational metrics (inference-time)

These are computed during inference and Phase 2, not Phase 1. They are the ground
truth signal for whether pretraining succeeded.

#### Hit rate by tier

```
L1 hit rate = segments served from L1 / total segments requested
L2 hit rate = segments served from L2 (no reconstruction) / total requests
L3 hit rate = segments reconstructed from L3 / total requests
Total miss rate = segments not found at any tier / total requests
```

**Interpretation for pretraining quality:**
- If reconstruction failure rate is high → L_recon did not converge
- If L2 hit rate is low despite segments being present → fingerprint mismatch;
  CE space doesn't preserve the semantic fingerprint signal
- If L3 misses are high → L3 CE (5 slots) compresses too aggressively; increase
  `C_L3` or decrease compression depth

#### Reconstruction confidence scores

Per-slot cosine similarity to stored anchor targets:
```
entity_slot_confidence    = cosine(CE[0:E], stored_entity_embeds)
boundary_slot_confidence  = cosine(CE[E:E+B], stored_boundary_ce)
semantic_slot_confidence  = cosine(CE[E+B:], semantic_fingerprint_mean)
```

**Targets at convergence:**
- Entity slots: mean > 0.7 (matches `tau=0.7` check threshold)
- Boundary slots: mean > 0.8 (directly initialized; should be very high)
- Semantic slots: mean > 0.5 (matches `reconstruction_theta=0.5`)

#### Fingerprint gate pass rate

How often reconstructed CEs pass the semantic fingerprint check
(`cosine(output_mean, semantic_fingerprint) ≥ 0.5`).

**Target:** > 90% pass rate. Below 70% indicates that compression is discarding the
semantic fingerprint signal and reconstruction cannot be trusted.

#### Entity fallback rate

How often the entity check fails all 3 retries and falls back to directly
initializing from stored entity embeddings.

**Target:** < 10%. Above 20% means the LoRA has not learned to route entity
information into the entity region.

### 3.3 Per-slot structure metrics (advanced)

These are not computed today but are high-value additions for understanding
CE quality:

**Slot utilization** — variance per slot across a batch. A slot with near-zero
variance is dead (always encoding the same vector regardless of input).
Target: all slots have variance > 0.1.

**CE centroid drift** — L2 distance of the mean CE vector over training time.
Sudden drift indicates a phase transition in the LoRA representation.
Large drift during Phase 2 fine-tuning indicates catastrophic forgetting.

**Slot-position correlation** — does the compressor consistently route the same
semantic type of information to the same slot position? This should emerge
naturally from the structured CE layout enforcement.

---

## 4. Key signals to watch

### 4.1 Healthy training trajectory

```
Step    L_distill   L_recon   Entity fallback   Reconstruct pass
  100     1.1–1.4    0.3–0.4         —                  —
  500     0.7–1.0    0.2–0.3         —                  —
 2000     0.4–0.6    0.15–0.2      20–30%            75–85%
10000     0.2–0.3    0.1–0.15      10–15%            88–92%
50000     0.1–0.15   0.05–0.1       <10%             >93%
```

### 4.2 Warning signs and responses

**L_distill not decreasing after 2,000 steps**
- Likely causes: learning rate too high, LoRA rank too low, data too short
- Check: plot per-layer distillation loss — if only first half of layers are
  improving, the CE may be optimizing for early layers only
- Fix: increase `lora_rank` (try 32), reduce `pretrain_learning_rate` to 1e-4,
  ensure documents are ≥ 50 segments

**L_recon > 0.3 after 5,000 steps**
- Likely causes: mean collapse (short repeated data), or entity/boundary regions
  occupying too many slots relative to semantic content
- Check: verify data is not the fallback "quick brown fox" string
- Fix: check data path; try increasing `C_L2` to 10 while keeping `C_L3=5`

**Entity fallback rate > 30% after 10,000 steps**
- Likely causes: entity-sparse training data, or `E=2` slots insufficient
- Fix: increase Wikipedia proportion in dataset mix; ensure spaCy NER is finding
  entities (check `entity_extractor.py` output on sample documents)

**Reconstruction gate pass rate dropping mid-training**
- Likely causes: L_recon is overfitting to mean alignment while losing structural
  variance; or fingerprinter MiniLM embeddings are inconsistent with CE space
- Fix: add L_inject as a regularizer; check that `reconstruction_theta` is calibrated
  to current distribution (may need recalibration after 50K steps)

**Phase 2 loss higher than Phase 1 baseline**
- Expected behavior: Phase 2 loss should be lower, since it operates on the specific
  segments that were recently missed
- If higher: the single-step update is overshooting; reduce `finetune_learning_rate`
  to 1e-5; or the missed segments are structurally different from pretraining
  distribution

**CE centroid drift during Phase 2**
- Indicates catastrophic forgetting of Phase 1 representations
- Fix: implement EWC regularization on LoRA parameters (deferred in current MVP);
  or freeze the compressor during Phase 2 and only fine-tune the reconstructor

### 4.3 Diagnostic commands

```bash
# Inspect compression output on a specific text snippet
python -c "
from utils.config import Config
from compression.compressor import Compressor
from transformers import AutoTokenizer
import torch

cfg = Config.load()
tok = AutoTokenizer.from_pretrained(cfg.model_name)
from transformers import AutoModelForCausalLM
llm = AutoModelForCausalLM.from_pretrained(cfg.model_name)
embed = llm.get_input_embeddings()

text = 'Alice met Bob in New York. They discussed the merger.'
ids = torch.tensor([tok.encode(text)])
orig = embed(ids)[0]

comp = Compressor(cfg, device='cpu')
ce = comp.compress(orig, cfg.C_L2)
print('CE shape:', ce.shape)
print('CE mean norm:', ce.mean(dim=0).norm().item())
print('CE slot norms:', ce.norm(dim=-1).tolist())
"
```

---

## 5. Observability infrastructure

### 5.1 Current state

The existing system logs to Python's `logging` module with INFO-level messages from
`cache_controller` and `inference_loop`. Training emits a single print line every
50 steps. This is functional for local development but insufficient for understanding
training dynamics.

### 5.2 Recommended observability stack

#### Training-time (Phase 1 + Phase 2)

Add Weights & Biases (or TensorBoard) instrumentation to `pretrain.py`:

```python
# At the top of train():
import wandb
wandb.init(
    project="semantic-forgetfulness",
    config={
        "lora_rank": cfg.lora_rank,
        "C_L2": cfg.C_L2,
        "C_L3": cfg.C_L3,
        "E": cfg.E,
        "B": cfg.B,
        "lr": cfg.pretrain_learning_rate,
        "steps": steps,
        "model": cfg.model_name,
    }
)

# In the training loop, replace the print with:
if step % 10 == 0:
    wandb.log({
        "loss/total": loss.item(),
        "loss/l_distill": l_distill.item(),
        "loss/l_recon": l_recon.item(),
        "grad_norm/compressor": _grad_norm(compressor),
        "grad_norm/reconstructor": _grad_norm(reconstructor),
        "ce/slot_norm_mean": ce_l2.norm(dim=-1).mean().item(),
        "ce/slot_norm_std": ce_l2.norm(dim=-1).std().item(),
    }, step=step)
```

Helper for gradient norm:
```python
def _grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return total ** 0.5
```

#### Per-layer distillation breakdown

The current code averages L_distill across all mid-to-late layers. Log per-layer
to identify which layers converge first:

```python
layer_losses = {}
for layer_idx in layer_range:
    h_full = out_full.hidden_states[layer_idx][:, -1, :]
    h_ce   = out_ce.hidden_states[layer_idx][:, -1, :]
    std    = h_full.std().clamp(min=1e-6)
    ll     = F.smooth_l1_loss(h_ce / std, h_full.detach() / std)
    layer_losses[f"l_distill/layer_{layer_idx}"] = ll.item()
    l_distill = l_distill + ll

wandb.log(layer_losses, step=step)
```

This reveals whether the CE is aligning in early, middle, or late layers — an
important architectural signal.

#### Reconstruction quality metrics

After each L_recon computation, log reconstruction diagnostics:

```python
if step % 50 == 0:
    result = reconstructor.reconstruct(ce_l2, anchors, [], None, "l3_to_l2")
    confidence_scores = [s for _, s in result.confidence_scores]
    wandb.log({
        "reconstruction/fingerprint_sim": result.fingerprint_sim,
        "reconstruction/confidence_mean": sum(confidence_scores) / len(confidence_scores),
        "reconstruction/confidence_min": min(confidence_scores),
        "reconstruction/fallback_rate": float(result.fallback),
        "reconstruction/grounding_used": float(result.grounding_used),
    }, step=step)
```

#### Inference-time metrics (session observability)

Add structured logging to `cache_controller.py` for real-time session health:

```python
# In cache_controller.py, track per-session aggregates:
self._session_stats = {
    "total_admits": 0,
    "l2_misses": 0,
    "l3_misses": 0,
    "total_misses": 0,
    "reconstruction_passes": 0,
    "reconstruction_failures": 0,
    "entity_fallbacks": 0,
    "promotions_to_l1": 0,
}

# At session end (or on /stats command), emit as structured log:
import json
logger.info(json.dumps({"event": "session_stats", **self._session_stats}))
```

Parse these logs with a simple script or forward to a metrics backend.

### 5.3 Dashboard design

For a W&B dashboard, organize panels into three sections:

**Training Health**
- `loss/total` over steps (line chart)
- `loss/l_distill` vs `loss/l_recon` (dual-axis)
- `grad_norm/compressor` and `grad_norm/reconstructor` (should stay in [0.01, 10])
- `ce/slot_norm_mean` ± `ce/slot_norm_std` (band chart; should stabilize)

**CE Quality**
- `reconstruction/fingerprint_sim` over steps (should rise toward 0.7+)
- `reconstruction/confidence_mean` by region (entity, boundary, semantic separately)
- `reconstruction/fallback_rate` (should fall below 0.1)
- Per-layer L_distill heatmap (layer × step; each cell is L_distill for that layer)

**Session Health (inference-time)**
- L1/L2/L3 hit rate as function of conversation turn
- Miss rate over conversation length (expect rise as L2/L3 fill and age)
- Reconstruction pass/fail ratio per session
- Phase 2 avg loss per session (should decrease across sessions)

### 5.4 Alerting thresholds

These are signals that warrant immediate investigation:

| Metric | Alert threshold | Likely cause |
|---|---|---|
| `loss/l_distill` after 5K steps | > 0.8 | LoRA rank too low or LR too high |
| `grad_norm/compressor` | > 50 | Exploding gradients; add gradient clipping |
| `grad_norm/compressor` | < 0.001 | Vanishing gradients; check frozen backbone isn't leaking requires_grad |
| `reconstruction/fallback_rate` | > 0.3 after 10K steps | Entity region not learning |
| `reconstruction/fingerprint_sim` | < 0.4 at convergence | Fingerprint space mismatch; recalibrate `reconstruction_theta` |
| `ce/slot_norm_std` | < 0.01 | Dead slots; CE collapsing to mean |

### 5.5 Gradient clipping (recommended addition)

The current training loop has no gradient clipping. With LoRA adapters on a large
LLM, gradient spikes are common in early training. Add before `optimizer.step()`:

```python
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
```

---

## 6. Training protocol

### 6.1 Phase 1 schedule (recommended)

A three-stage curriculum improves convergence vs. random mixing:

**Stage 1: Entity anchor stabilization (0–10K steps)**
- Data: 70% Wikipedia, 30% OpenWebText2
- Goal: Entity and boundary regions stabilize before semantic content training
- Stop condition: Entity fallback rate < 20%, L_recon < 0.25
- LR: `pretrain_learning_rate` default (2e-4)

**Stage 2: Semantic content pretraining (10K–100K steps)**
- Data: Full recommended mix (40/25/20/10/5 split)
- Goal: L_distill and L_recon converge to target ranges
- Stop condition: L_distill < 0.15, L_recon < 0.1, fingerprint pass rate > 90%
- LR: 1e-4 (halved from stage 1)

**Stage 3: Task alignment (100K–150K steps, optional)**
- Data: NarrativeQA QA pairs + UltraChat for L_inject signal
- Goal: Verify CE injection does not degrade downstream task quality
- Stop condition: Δ task perplexity < 0.05
- LR: 5e-5

### 6.2 Checkpoint evaluation protocol

Every 10K steps (or before deployment), run the following evaluations:

**1. Compression round-trip test**
```bash
python -c "
# Load checkpoint, compress a held-out segment, reconstruct,
# measure cosine similarity to original. Target > 0.85.
"
```

**2. LLM injection coherence test**
Run a 20-turn conversation on a held-out topic. Check:
- `/stats` at turn 10: L2 should have content
- `/stats` at turn 20: L3 should have content
- Ask a question about information from turn 1. If answered correctly: injection works.

**3. Phase 2 responsiveness test**
Run a session with known-miss segments (use a topic from the test set, then ask
about it from a different angle to trigger misses). Verify Phase 2 fires and
`avg_loss` decreases vs. the same session pre-fine-tuning.

### 6.3 Phase 2 monitoring per session

Each session end should log:

```
[Fine-tuning] Events: 23  Avg loss: 0.1823  Steps: 1  LR: 5e-05
[Fine-tuning] Checkpoint saved: checkpoints/ (compressor + reconstructor)
```

Track `avg_loss` across sessions. It should decrease as the adapters specialize
to the conversation domain. A sudden increase indicates distribution shift (the
conversation covered an entirely new domain) — which is the intended behavior.

---

## 7. Scaling the data pipeline

The current `pretrain.py` reads a single `.txt` file. For multi-dataset training
at 10M+ tokens, replace this with a streaming `datasets` loader:

```python
# In pretrain.py, replace the text loading block with:
from datasets import load_dataset, interleave_datasets

wiki = load_dataset("wikimedia/wikipedia", "20220301.en",
                    split="train", streaming=True)
owt  = load_dataset("EleutherAI/pile", data_files="*openwebtext2*",
                    split="train", streaming=True)
chat = load_dataset("HuggingFaceH4/ultrachat_200k",
                    split="train_sft", streaming=True)

# Mix with weights
dataset = interleave_datasets(
    [wiki, owt, chat],
    probabilities=[0.4, 0.4, 0.2],
    stopping_strategy="all_exhausted",
)

# Extract text field (varies by dataset)
def get_text(example):
    return example.get("text") or example.get("content") or ""

# Replace the text/segments generation loop
for raw in dataset:
    doc_text = get_text(raw)
    if len(doc_text) < 500:
        continue  # skip very short documents
    all_ids = tokenizer.encode(doc_text, add_special_tokens=False)
    segments = segmenter.segment(all_ids)
    for seg_ids in segments:
        # ... existing training step
```

This requires `pip install datasets` (already a dependency for Hugging Face models).

---

## 8. L2→L3 compression pretraining (advanced)

The current `pretrain.py` only trains L1→L2 compression (20 tokens → 8 CE slots).
The L2→L3 path (8 CE slots → 5 CE slots) is also trainable but requires a
different data generation approach, since the "input" is a CE tensor, not token
embeddings.

**Bootstrap procedure:**
1. Run Stage 1+2 pretraining to convergence (train L1→L2 first)
2. Generate an L2 CE dataset: run the trained compressor on all training segments,
   save the `(original_tokens, L2_CE)` pairs
3. Run L2→L3 training: input = L2_CE, target = original_tokens
   (same L_recon objective; L_distill omitted per design)
4. Fine-tune jointly for the final 10K steps

This two-stage bootstrap is the correct order because L2→L3 training requires
high-quality L2 CEs as inputs — training on random-init L2 CEs produces garbage
L3 representations.

---

## 9. Quick reference: target metric table

| Metric | Healthy at 10K steps | Excellent at 50K steps | Alert threshold |
|---|---|---|---|
| `L_distill` | 0.3–0.5 | < 0.15 | > 0.8 after 5K |
| `L_recon` | 0.15–0.25 | < 0.1 | > 0.4 after 5K |
| Entity fallback rate | 15–25% | < 10% | > 30% after 10K |
| Fingerprint pass rate | 80–88% | > 93% | < 70% |
| Confidence mean (entity slots) | 0.55–0.65 | > 0.7 | < 0.4 |
| Confidence mean (boundary slots) | 0.7–0.8 | > 0.85 | < 0.5 |
| Confidence mean (semantic slots) | 0.4–0.55 | > 0.6 | < 0.35 |
| CE slot norm std | > 0.1 | > 0.2 | < 0.01 (dead slot) |
| Grad norm (compressor) | 0.1–5 | 0.05–2 | > 50 or < 0.001 |
| Phase 2 avg_loss | < 0.25 | < 0.1 | > Phase 1 baseline |
| L2 reconstruction pass rate | > 80% | > 93% | < 70% |
| Inference hit rate (session L1) | > 85% | > 95% | < 70% |
