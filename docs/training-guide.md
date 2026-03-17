# Semantic Forgetfulness — Training Guide

A complete, step-by-step walkthrough for training the system from scratch.
No prior familiarity with the codebase is assumed.

---

## What you are training

Semantic Forgetfulness is a memory compression system for large language models.
It learns to compress conversation segments into compact **Concept Embeddings (CEs)**
that the frozen LLM can later use to reconstruct the original context.

Two LoRA adapters are trained (both applied on top of the same frozen LLM backbone):

| Adapter | Job |
|---|---|
| **Compressor** | Takes a segment's token embeddings `[N, D]` → produces `C` CE slots `[C, D]` |
| **Reconstructor** | Takes CE slots → produces a higher-fidelity CE ready for KV injection |

Training happens in two phases:

- **Phase 1 — Offline pretraining** on a static text corpus. Teaches the adapters the basic compression task.
- **Phase 2 — Online fine-tuning** at the end of each live session. Adapts the adapters to segments that were actually retrieved (cache misses) during that conversation.

---

## Prerequisites

### 1. Conda environment

The project uses a dedicated conda environment called `sf-mvp` with Python 3.11.
If it does not exist yet, create and install it:

```bash
conda create -n sf-mvp python=3.11 -y
conda activate sf-mvp
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

Verify the environment works:

```bash
conda activate sf-mvp
pytest tests/ -q
# Expected: 42 passed
```

### 2. Hugging Face access (for Llama)

The default model is `meta-llama/Llama-3.2-3B-Instruct`, which is gated on Hugging Face.
You must:

1. Create a Hugging Face account at huggingface.co
2. Go to the model page and click **Request Access**
3. Once approved, generate an access token at huggingface.co/settings/tokens
4. Log in locally:

```bash
conda activate sf-mvp
hf auth login
# Paste your token when prompted
```

> **Skip this if using GPT-2.** GPT-2 is ungated. See the section below on using an alternate model.

---

## Phase 1 — Offline pretraining

### What this does

Pretraining teaches the Compressor and Reconstructor to produce CEs that carry
the same semantic signal as the original token sequence. It does this with two losses:

- **L_distill** — The LLM's hidden states when processing the CE slots should match
  the hidden states when processing the full token sequence, at mid-to-late layers.
  This forces CEs to be semantically equivalent to the original from the LLM's point of view.
- **L_recon** — The mean of the reconstructed CE should have high cosine similarity
  to the mean of the original token embeddings. This ensures the reconstruction
  is pointed in the right direction.

### Prepare training data

Create a file `data/train.txt` at the repo root containing plain text.
Good sources: books, Wikipedia dumps, conversation transcripts, domain-specific text.
The more text the better — aim for at least 50,000 words for a meaningful run.

```bash
mkdir -p data
# Copy or write your text file here:
# data/train.txt
```

> **No data file?** If `data/train.txt` is missing, the trainer falls back to a short
> repeated sentence ("The quick brown fox..."). This is only useful for verifying the
> pipeline runs, not for learning anything meaningful.

### Run pretraining

```bash
conda activate sf-mvp
python -m sf.training.pretrain \
  --data-path data/train.txt \
  --steps 500 \
  --device cpu
```

Replace `--device cpu` with `--device cuda` if you have a GPU.

**Arguments:**

| Argument | Default | Meaning |
|---|---|---|
| `--data-path` | `data/train.txt` | Path to training text |
| `--steps` | `500` | Gradient steps to run |
| `--device` | `cpu` | `cpu` or `cuda` |

**What you will see:**

```
Step 50/500  L_distill=0.8432  L_recon=0.2341
Step 100/500  L_distill=0.6201  L_recon=0.1987
...
Adapters saved to checkpoints/
```

Both losses should decrease over time. `L_distill` starts higher (often 0.5–1.5)
and should trend toward 0. `L_recon` should stay below 0.5 and decrease.

### Output

Pretraining saves two LoRA adapter checkpoints:

```
checkpoints/
  compressor/      ← Compressor LoRA weights
  reconstructor/   ← Reconstructor LoRA weights
```

These are standard Hugging Face PEFT adapter directories and can be loaded with
`PeftModel.from_pretrained(...)`.

### Tuning pretrain hyperparameters

Edit `config.yaml` at the repo root:

```yaml
lora_rank: 16                  # LoRA rank — higher = more capacity, slower
pretrain_learning_rate: 0.0002 # AdamW learning rate for pretraining
C_L2: 8                        # Number of CE slots at L2 (compression ratio)
C_L3: 5                        # Number of CE slots at L3 (more compressed)
```

> **CE layout constraint:** `C_L2 > C_L3 > E + B` must always hold.
> With defaults: `8 > 5 > 4`. Violating this raises an error on startup.

---

## Phase 2 — Online fine-tuning (per session)

### What this does

After a live conversation session ends, the system fine-tunes the adapters on the
segments that triggered **cache misses** — segments the model failed to retrieve
correctly from L2 or L3. This specialises the compressor to the topics actually
discussed in the session.

Fine-tuning uses a single gradient step (configurable) over the missed segments,
minimising cosine distance between the compressed CE and the original embedding mean.

Phase 2 is triggered automatically at the end of a chatbot session. It runs inside
the same process — no separate command needed.

### Requirements for fine-tuning to run

Fine-tuning is **skipped** if fewer than `min_session_length` miss events were collected
(default: 20). This prevents overfitting on tiny sessions. You must have a long enough
conversation with enough cache misses before fine-tuning fires.

Fine-tuning also requires `--load-models` to be active (mock mode has no adapters).

---

## Running a full training loop

A complete training experiment looks like this:

### Step 1: Run Phase 1 pretraining

```bash
conda activate sf-mvp
python -m sf.training.pretrain --data-path data/train.txt --steps 500 --device cpu
```

Wait for "Adapters saved to checkpoints/".

### Step 2: Start a chatbot session

```bash
python main.py --load-models --device cpu
```

For Llama (gated):
```bash
python main.py --load-models --device cpu
```

For GPT-2 (no HF login needed, smaller model, faster):
```bash
python main.py --model gpt2 --load-models --device cpu
```

You will see:
```
Semantic Forgetfulness
  Model : meta-llama/Llama-3.2-3B-Instruct
  Device: cpu  |  Full inference: True
  Commands: /stats  /done  /quit
```

### Step 3: Have a conversation

Type messages and read responses. The cache hierarchy is actively building in the background.
Check how the memory tiers are filling:

```
You: /stats
```

Output:
```
── Cache Stats ──────────────────────────────
  Segments  : 12  (L1=12 L2=0 L3=0)
  L1 rate   : 100.0%
  Misses    : 0
─────────────────────────────────────────────
```

- **L1** fills first (fast token buffer). L2 activates after 8,000 tokens, L3 after 32,000.
- **Misses** are segments the system tried to retrieve from L2/L3 but found semantically stale.

### Step 4: End the session and trigger Phase 2

```
You: /done
```

The session ends, miss events are collected, and Phase 2 fine-tuning runs:

```
[Session ending...]
[Fine-tuning: 23 events, loss=0.1823]
[Adapters saved to checkpoints/]
```

If you see `[Fine-tuning skipped: insufficient_miss_events]`, the session was too short
(fewer than 20 cache misses). Have a longer conversation, or lower `min_session_length`
in `config.yaml`.

---

## Using GPT-2 instead of Llama

GPT-2 is ungated, much smaller (117M vs 3B parameters), and useful for quick experiments.
The tradeoff is lower response quality.

When using GPT-2, you must also update `config.yaml` to match its embedding dimension:

```yaml
model_name: "gpt2"
embed_dim: 768   # GPT-2 hidden size (Llama-3.2-3B is 3072)
```

Then run:

```bash
python main.py --model gpt2 --load-models
```

Or for pretraining:

```bash
python -m sf.training.pretrain --data-path data/train.txt --steps 200 --device cpu
```

> **Remember to revert `embed_dim` back to 3072** if you switch back to Llama.

---

## Mock mode (no model download)

Mock mode runs the full pipeline without loading any LLM. Useful for verifying the
cache hierarchy, segmenter, and statistics work without needing a GPU or HF token.

```bash
python main.py
```

Responses will be `[mock response — run with load_models=True for real inference]`.
All cache mechanics (segment admission, eviction, miss logging, `/stats`) are live.
Fine-tuning is skipped at session end in mock mode.

---

## Key hyperparameters reference

All hyperparameters live in `config.yaml`. Changes take effect on the next run — no recompilation needed.

| Parameter | Default | What it controls |
|---|---|---|
| `model_name` | `meta-llama/Llama-3.2-3B-Instruct` | Base LLM |
| `embed_dim` | `3072` | Must match the LLM's hidden size |
| `C_L2` | `8` | CE slots at L2 (compression ratio: 20 tokens → 8 CE slots) |
| `C_L3` | `5` | CE slots at L3 (further compressed) |
| `E` | `2` | Entity anchor slots per CE |
| `B` | `2` | Boundary anchor slots per CE |
| `l2_capacity` | `200` | Max segments held in L2 |
| `l3_capacity` | `1000` | Max segments held in L3 |
| `l2_activation_threshold` | `8000` | Total tokens before L2 tier activates |
| `l3_activation_threshold` | `32000` | Total tokens before L3 tier activates |
| `miss_detection_theta` | `0.7` | Cosine similarity floor for cache hit (0–1, higher = stricter) |
| `leniency` | `2` | Faults before a miss triggers L1 promotion |
| `lora_rank` | `16` | LoRA adapter rank (higher = more capacity) |
| `pretrain_learning_rate` | `0.0002` | AdamW LR for Phase 1 |
| `finetune_learning_rate` | `0.00005` | AdamW LR for Phase 2 |
| `min_session_length` | `20` | Minimum miss events before Phase 2 runs |
| `fine_tuning_steps` | `1` | Gradient steps per session end |

---

## Checkpoint management

Checkpoints are saved to `checkpoints/` at the repo root after both Phase 1 and Phase 2.

```
checkpoints/
  compressor/
    adapter_config.json
    adapter_model.safetensors
  reconstructor/
    adapter_config.json
    adapter_model.safetensors
```

To load saved adapters in your own code:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
compressor_model = PeftModel.from_pretrained(base, "checkpoints/compressor")
```

> Checkpoints are **overwritten** each time fine-tuning runs. If you want to preserve
> a checkpoint, copy the `checkpoints/` directory before the next session ends.

---

## Running tests

To verify nothing is broken after config changes:

```bash
conda activate sf-mvp
pytest tests/ -q
```

Expected output: `42 passed`. Any failure indicates a regression.

For verbose output with which tests ran:

```bash
pytest tests/ -v
```
