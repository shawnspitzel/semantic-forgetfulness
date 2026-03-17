# Hyperparameters

> All tunable parameters across the Semantic Forgetfulness system, with defaults and rationale. Parameters marked *empirical* require measurement on the target model and task distribution before finalizing.

---

## Cache Controller

| Parameter | Default | Notes |
|-----------|---------|-------|
| `alpha` | 0.6 | Importance weight in eviction score: `alpha * importance + (1-alpha) * recency`. |
| `lambda` | 0.1 | Recency decay rate: `exp(-lambda * delta_t)`. Higher = recency decays faster. |
| `leniency` | 2 | Miss count before a segment is promoted to a higher tier. 1 = promote on first miss. Starting point from Pichay et al. |
| `miss_detection_theta` | 0.7 | Cosine similarity threshold (query vs. segment fingerprint) for a segment to be considered "needed." |
| `l2_capacity` | *empirical* | Max segments in L2. Controls warm-start conservatism — L2→L3 eviction cannot fire until L2 is full. |
| `l3_capacity` | *empirical* | Max segments in L3. Determines total session memory budget. |
| `l2_activation_threshold` | 8,000 tokens | Context length at which L2 engages. Below this, compression overhead is not justified. |
| `l3_activation_threshold` | 32,000 tokens | Context length at which L3 engages. |

---

## Compressor

| Parameter | Default | Notes |
|-----------|---------|-------|
| `C_L2` | 10 slots | CE slots per ~20-token segment for L1→L2 (~2x compression). CompLLM baseline. Must satisfy `C_L2 > C_L3 > E + B`. |
| `C_L3` | 5 slots | CE slots per L2 entry for L2→L3 (~4x total from original). Must satisfy `C_L3 > E + B`. |
| `E` (max_entities) | 4 | Entity anchor slots in CE layout (CE[0:E]). Fixed before training. Typical segments have 1–4 named entities. |
| `B` | 2 | Boundary slots in CE layout (CE[E:E+B]). Fixed — not tunable. |
| `lora_rank` | 16 | LoRA adapter rank. Controls trainable parameter count. |
| `distill_layer_range` | *empirical* | LLM layers at which L_distill is measured. Mid-to-late depth per CompLLM. Model-family-dependent. |

---

## Reconstructor

| Parameter | Default | Notes |
|-----------|---------|-------|
| `reconstruction_theta` | 0.75 | Cosine similarity threshold between reconstructed output and stored semantic fingerprint. Output below this is rejected. Midpoint of 0.7–0.8 empirical range. |
| `tau` | *empirical* | Cosine similarity threshold for per-entity-slot validation during reconstruction. |
| `reconstruction_retry_budget` | 3 | Max retries before falling back to boundary sentences + entity list only. |

---

## Inference Loop

| Parameter | Default | Notes |
|-----------|---------|-------|
| `tsp_layer_index` | `floor(num_layers / 2)` | Layer used for attention concentration importance scoring. FastKV mid-depth guidance. Model-family-dependent — tune empirically. |
| `segment_length` | 20 tokens | Target segment size (sentence-boundary aligned). From CompLLM. |
| `segment_hard_cap` | 25 tokens | Max segment size before a hard boundary is forced. |

---

## Training — Phase 1 (Distillation Pretraining)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `pretrain_learning_rate` | *empirical* | Standard LoRA starting range: 1e-4 to 3e-4. |
| `pretrain_batch_size` | *empirical* | Batch size for initial distillation training. |
| `pretrain_dataset` | NarrativeQA + LOFT | Aligned with benchmark evaluation distribution per answers.md P9.2. |

---

## Training — Phase 2 (End-of-Session Online Learning)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `min_session_length` | 20 | Minimum cache miss events before end-of-session weight update fires. Below this, skip the update — too few misses for a stable gradient. |
| `finetune_learning_rate` | *empirical* | Expected smaller than pretrain_learning_rate — these are refinement steps, not initial training. |
| `fine_tuning_steps` | 1 | Full passes over miss_log before the optimizer step. |
| `context_window_W` | 100 tokens | Token window stored around each segment at admission time for teacher re-run at session end. |
