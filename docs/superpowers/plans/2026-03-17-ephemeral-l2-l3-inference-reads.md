# Ephemeral L2/L3 Inference Reads Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Include reconstructed L2 and L3 content in `inputs_embeds` at inference time, regardless of whether promotion is triggered, so the model always has access to all remembered context (lossily).

**Architecture:** Add a `reconstruct_for_inference` method to `CacheController` that reconstructs L2/L3 segments ephemerally (no tier mutation, no L1 insert). Modify `generate_response` in `InferenceLoop` to reconstruct all misses, merge those embeddings into `inputs_embeds` (ordered by `source_position`), and separately handle promotion as a persistence decision. Promotion remains gated by `fault_count >= leniency`.

**Tech Stack:** Python, PyTorch, existing `CacheController`, `L1Entry`, `Reconstructor`

---

## File Map

| File | Change |
|---|---|
| `src/memory/cache_controller.py` | Add `reconstruct_for_inference(segment_id, query_vec)` |
| `src/inference/inference_loop.py` | Modify `generate_response` to collect ephemeral entries, merge into `inputs_embeds` by source position, update logging |
| `tests/conftest.py` | Add missing `memory_enabled=True` to cfg fixture |
| `tests/test_cache_controller.py` | Tests for `reconstruct_for_inference` |
| `tests/test_inference_loop.py` | Test that ephemeral reconstructions appear in context |

---

### Task 0: Fix missing `memory_enabled` in conftest

**Files:**
- Modify: `tests/conftest.py`

`Config` requires `memory_enabled: bool` but the test fixture omits it, causing `TypeError` for any test that instantiates a real `Config`. Fix before adding new tests.

- [ ] **Step 1: Add `memory_enabled=True` to the cfg fixture**

In `tests/conftest.py`, add `memory_enabled=True` inside the `Config(...)` call:

```python
@pytest.fixture(scope="session")
def cfg():
    """Test config — uses gpt2 (small, no gated access) and CPU-friendly sizes."""
    return Config(
        model_name="gpt2",
        embed_dim=768,
        segment_target=10,
        segment_hard_cap=15,
        C_L2=8,
        C_L3=5,
        E=2,
        B=2,
        l2_capacity=10,
        l3_capacity=20,
        l2_activation_threshold=50,
        l3_activation_threshold=100,
        alpha=0.6,
        lam=0.1,
        miss_detection_theta=0.7,
        leniency=2,
        reconstruction_theta=0.75,
        tau=0.7,
        reconstruction_retry_budget=3,
        tsp_layer_index=-1,
        memory_enabled=True,   # <-- add this
        lora_rank=16,
        min_session_length=20,
        fine_tuning_steps=1,
        context_window_W=100,
        pretrain_learning_rate=2e-4,
        finetune_learning_rate=5e-5,
    )
```

- [ ] **Step 2: Run existing tests to confirm no regressions**

```bash
cd c:/Users/kevin/semantic-forgetfulness
python -m pytest tests/ -v --tb=short
```
Expected: all existing tests still pass.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "fix: add missing memory_enabled=True to test cfg fixture"
```

---

### Task 1: Extract `reconstruct_segment` and simplify `promote_to_l1`

**Files:**
- Modify: `src/memory/cache_controller.py`
- Test: `tests/test_cache_controller.py`

Extract shared reconstruction logic into a public `reconstruct_segment` method with no side effects. Refactor `promote_to_l1` to call it and handle only the persistence step. The inference loop (Task 2) calls `reconstruct_segment` directly — no wrapper method needed.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cache_controller.py` (extend existing imports, do NOT replace them):

```python
# Extend existing imports:
import time  # already present
from utils.data_structures import L2Entry, ReconstructionResult  # add alongside SanityAnchors


def _mock_reconstruct(ce_tensor, anchors, neighbors, query_vec, stage):
    """Stub reconstruct_fn: echoes CE tensor back with fingerprint_sim=1.0."""
    return ReconstructionResult(
        ce_tensor=ce_tensor,
        confidence_scores=[(i, 1.0) for i in range(ce_tensor.shape[0])],
        fingerprint_sim=1.0,
        grounding_used=False,
        fallback=False,
    )


def _inject_l2(cc, sid, fp, cfg):
    """Helper: inject a fake L2 entry and mark metadata tier as l2."""
    cc._l2._entries[sid] = L2Entry(
        id=sid, ce_tensor=torch.randn(cfg.C_L2, cfg.embed_dim), semantic_fingerprint=fp,
        confidence_scores=None, importance_score=0.5,
        last_accessed=time.time(), source_position=0,
        session_id="s1", origin="l1_demotion", grounding_used=False, entities=[],
    )
    cc._metadata[sid].tier = "l2"


def test_reconstruct_segment_returns_entry_without_side_effects(cfg):
    """reconstruct_segment builds an L1Entry but leaves tier and L1 store unchanged."""
    cc = CacheController(cfg=cfg, session_id="s1", reconstruct_fn=_mock_reconstruct)
    sid = uuid.uuid4()
    fp = F.normalize(torch.randn(768), dim=0)
    cc.admit(sid, torch.randn(20, cfg.embed_dim), 0.5, 0, fp,
             SanityAnchors(["First.", "Last."], [], fp), total_tokens_seen=0)
    _inject_l2(cc, sid, fp, cfg)

    entry = cc.reconstruct_segment(sid, fp)

    assert entry is not None
    assert entry.is_reconstructed is True
    assert cc._metadata[sid].tier == "l2", "Tier must NOT change"
    assert sid not in cc._l1._entries, "Must NOT be inserted into L1 store"


def test_reconstruct_segment_returns_none_without_reconstruct_fn(cfg):
    """Returns None gracefully when no reconstruct_fn is set."""
    cc = CacheController(cfg=cfg, session_id="s1")
    sid = uuid.uuid4()
    fp = F.normalize(torch.randn(768), dim=0)
    cc.admit(sid, torch.randn(20, cfg.embed_dim), 0.5, 0, fp,
             SanityAnchors(["First.", "Last."], [], fp), total_tokens_seen=0)
    cc._metadata[sid].tier = "l2"

    assert cc.reconstruct_segment(sid, fp) is None


def test_promote_to_l1_delegates_to_reconstruct_segment(cfg):
    """promote_to_l1 still promotes correctly after refactor."""
    cc = CacheController(cfg=cfg, session_id="s1", reconstruct_fn=_mock_reconstruct)
    sid = uuid.uuid4()
    fp = F.normalize(torch.randn(768), dim=0)
    cc.admit(sid, torch.randn(20, cfg.embed_dim), 0.5, 0, fp,
             SanityAnchors(["First.", "Last."], [], fp), total_tokens_seen=0)
    _inject_l2(cc, sid, fp, cfg)

    result = cc.promote_to_l1(sid, fp)

    assert result is not None
    assert result.is_reconstructed is True
    assert cc._metadata[sid].tier == "l1", "Tier MUST change on promotion"
    assert cc._metadata[sid].fault_count == 0, "fault_count MUST reset on promotion"
    assert sid in cc._l1._entries, "MUST be inserted into L1 store"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_cache_controller.py::test_reconstruct_segment_returns_entry_without_side_effects tests/test_cache_controller.py::test_reconstruct_segment_returns_none_without_reconstruct_fn tests/test_cache_controller.py::test_promote_to_l1_delegates_to_reconstruct_segment -v
```
Expected: first two fail with `AttributeError: 'CacheController' object has no attribute 'reconstruct_segment'`; third may pass or fail.

- [ ] **Step 3: Refactor `cache_controller.py`**

Replace the existing `promote_to_l1` method with the following two methods:

```python
def reconstruct_segment(
    self, segment_id: uuid.UUID, query_vec: torch.Tensor
) -> Optional[L1Entry]:
    """
    Reconstruct an L2 or L3 segment into an L1Entry. No state is mutated.
    Returns None if reconstruct_fn is unavailable, the segment is missing,
    or reconstruction quality is below reconstruction_theta.
    """
    if self.reconstruct_fn is None:
        return None
    meta = self._metadata.get(segment_id)
    if meta is None:
        return None
    anchors = self._anchors.get(segment_id)

    if meta.tier == "l2":
        l2e = self._l2.get(segment_id)
        if not l2e:
            return None
        neighbors = self._l2.get_neighbors(segment_id)
        result = self.reconstruct_fn(l2e.ce_tensor, anchors, neighbors, query_vec, "l2_to_l1")
    elif meta.tier == "l3":
        l3e = self._l3.get(segment_id)
        if not l3e:
            return None
        neighbors = self._l2.get_neighbors(segment_id)
        r_l3 = self.reconstruct_fn(l3e.concept_embeddings, anchors, neighbors, None, "l3_to_l2")
        if r_l3.fingerprint_sim < self.cfg.reconstruction_theta:
            return None
        result = self.reconstruct_fn(r_l3.ce_tensor, anchors, neighbors, query_vec, "l2_to_l1")
    else:
        return None

    if result.fingerprint_sim < self.cfg.reconstruction_theta:
        return None

    return L1Entry(
        id=segment_id,
        tokens=torch.zeros(result.ce_tensor.shape[0], dtype=torch.long),
        token_embeddings=result.ce_tensor,
        importance_score=meta.importance_score,
        last_accessed=time.time(),
        source_position=meta.source_position,
        session_id=self.session_id,
        is_reconstructed=True,
    )

def promote_to_l1(self, segment_id: uuid.UUID, query_vec: torch.Tensor) -> Optional[L1Entry]:
    """Reconstruct and persist to L1 — mutates tier metadata and inserts into L1 store."""
    entry = self.reconstruct_segment(segment_id, query_vec)
    if entry is None:
        return None
    meta = self._metadata[segment_id]
    src_tier = meta.tier
    meta.tier = "l1"
    meta.fault_count = 0
    logger.info("[CC] Promoted  seg=%.8s  %s → L1", segment_id, src_tier.upper())
    self._l1.insert(entry)
    return entry
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_cache_controller.py::test_reconstruct_segment_returns_entry_without_side_effects tests/test_cache_controller.py::test_reconstruct_segment_returns_none_without_reconstruct_fn tests/test_cache_controller.py::test_promote_to_l1_delegates_to_reconstruct_segment -v
```
Expected: all three PASS

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
python -m pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/memory/cache_controller.py tests/test_cache_controller.py
git commit -m "refactor: extract reconstruct_segment; promote_to_l1 delegates to it"
```

---

### Task 2: Wire ephemeral reconstructions into `generate_response`

**Files:**
- Modify: `src/inference/inference_loop.py`
- Test: `tests/test_inference_loop.py`

The `generate_response` loop currently only collects `l1_entries()` for `inputs_embeds`. After this task, it will also:
1. Call `reconstruct_segment` for every detected miss
2. Merge those ephemeral entries with L1 entries, ordered by `source_position` for context coherence
3. Deduplicate: skip ephemeral entry for any segment that was promoted in the same pass (it's already in `l1_entries()`)
4. Still separately promote when `fault_count >= leniency`
5. Log ephemeral read count in `[Inference]` line

- [ ] **Step 1: Write the failing test**

Add to `tests/test_inference_loop.py`:

```python
import uuid
import time
import torch
import torch.nn.functional as F
from inference.inference_loop import InferenceLoop
from utils.data_structures import SanityAnchors, L2Entry, ReconstructionResult, SegmentMetadata


def _make_mock_reconstruct(embed_dim, C_L2):
    def _reconstruct(ce_tensor, anchors, neighbors, query_vec, stage):
        return ReconstructionResult(
            ce_tensor=torch.randn(C_L2, embed_dim),
            confidence_scores=[(i, 1.0) for i in range(C_L2)],
            fingerprint_sim=1.0,
            grounding_used=False,
            fallback=False,
        )
    return _reconstruct


def test_ephemeral_l2_reconstruction_included_in_context(cfg):
    """
    A segment in L2 should be reconstructed ephemerally and included in
    inputs_embeds during generate_response, even when fault_count < leniency.
    """
    loop = InferenceLoop(cfg)
    loop.cache_controller.reconstruct_fn = _make_mock_reconstruct(cfg.embed_dim, cfg.C_L2)

    # Use a deterministic fingerprint that will always exceed miss_detection_theta
    fp = F.normalize(torch.ones(768), dim=0)

    sid = uuid.uuid4()
    anchors = SanityAnchors(["First.", "Last."], ["Alice"], fp)
    fake_ce = torch.randn(cfg.C_L2, cfg.embed_dim)

    loop.cache_controller._anchors[sid] = anchors
    loop.cache_controller._metadata[sid] = SegmentMetadata(
        id=sid, tier="l2", importance_score=0.5,
        last_accessed=time.time(), source_position=0,
        session_id=loop.session_id, is_reconstructed=False,
        semantic_fingerprint=fp, fault_count=0, original_length=10,
    )
    loop.cache_controller._l2._entries[sid] = L2Entry(
        id=sid, ce_tensor=fake_ce, semantic_fingerprint=fp,
        confidence_scores=None, importance_score=0.5,
        last_accessed=time.time(), source_position=0,
        session_id=loop.session_id, origin="l1_demotion",
        grounding_used=False, entities=[],
    )

    # Capture inputs_embeds passed to generate
    captured = {}

    class MockEmbedding:
        def __call__(self, ids):
            return torch.randn(ids.shape[-1], cfg.embed_dim)

    class MockLLM:
        config = type("C", (), {"num_hidden_layers": 2})()
        def get_input_embeddings(self): return MockEmbedding()
        def generate(self, inputs_embeds=None, **kw):
            captured["inputs_embeds"] = inputs_embeds
            return torch.zeros(1, 1, dtype=torch.long)
        def disable_adapter(self):
            import contextlib; return contextlib.nullcontext()

    class MockTokenizer:
        eos_token_id = 0
        def encode(self, t, **kw): return [0] * 5
        def apply_chat_template(self, m, **kw): return torch.zeros(1, 3, dtype=torch.long)
        def decode(self, ids, **kw): return "ok"

    loop._llm = MockLLM()
    loop._tokenizer = MockTokenizer()
    loop.fingerprinter.encode = lambda text: fp  # always matches our L2 segment

    loop.generate_response("How old am I?")

    assert "inputs_embeds" in captured, "generate() was never called"
    seq_len = captured["inputs_embeds"].shape[1]
    # seq_len must be > query-only length (3 tokens from apply_chat_template mock)
    # because C_L2=8 ephemeral tokens should have been prepended
    assert seq_len > 3, (
        f"Expected ephemeral L2 tokens to widen inputs_embeds beyond query length, got {seq_len}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_inference_loop.py::test_ephemeral_l2_reconstruction_included_in_context -v
```
Expected: FAIL — `AssertionError: Expected ephemeral L2 tokens to widen inputs_embeds...`

- [ ] **Step 3: Replace the miss-handling block in `generate_response`**

In `src/inference/inference_loop.py`, replace lines ~193–216 (the `if self.cfg.memory_enabled:` block, up to and including the `inputs_embeds` construction) with:

```python
if self.cfg.memory_enabled:
    query_fp = self.fingerprinter.encode(query)
    misses = self.cache_controller.detect_misses(query_fp)
    promoted_from: dict[str, int] = {}
    promoted_ids: set = set()
    ephemeral_entries: list = []  # L1Entry objects not stored in L1

    for miss in misses:
        meta = self.cache_controller.get_metadata(miss.segment_id)
        if meta is None:
            continue

        # Always reconstruct for inference (ephemeral, lossy read from L2/L3)
        ephemeral = self.cache_controller.reconstruct_segment(
            miss.segment_id, query_fp
        )
        if ephemeral is not None:
            ephemeral_entries.append(ephemeral)

        # Separately: promote (persist to L1) if fault threshold reached
        if meta.fault_count >= self.cfg.leniency:
            src_tier = meta.tier
            result = self.cache_controller.promote_to_l1(miss.segment_id, query_fp)
            if result:
                promoted_from[src_tier] = promoted_from.get(src_tier, 0) + 1
                promoted_ids.add(miss.segment_id)

    # Exclude ephemeral entries for segments that were promoted
    # (they're already in l1_entries() after promotion)
    ephemeral_entries = [e for e in ephemeral_entries if e.id not in promoted_ids]

    emb_layer = self._llm.get_input_embeddings()
    l1_entries = self.cache_controller.l1_entries()
    native = sum(1 for e in l1_entries if not e.is_reconstructed)
    promoted_now = sum(promoted_from.values())
    l2_hits = promoted_from.get("l2", 0)
    l3_hits = promoted_from.get("l3", 0)
    logger.info(
        "[Inference] Reading L1=%d segs  native=%d  promoted=%d (from L2:%d L3:%d)"
        "  ephemeral=%d",
        len(l1_entries), native, promoted_now, l2_hits, l3_hits, len(ephemeral_entries),
    )

    # Merge L1 + ephemeral entries sorted by source_position for context coherence
    all_entries = sorted(
        l1_entries + ephemeral_entries,
        key=lambda e: e.source_position,
    )
    l1_embeds = [e.token_embeddings.to(self.device) for e in all_entries]
    query_embeds = emb_layer(formatted_ids)[0]
    inputs_embeds = torch.cat(l1_embeds + [query_embeds], dim=0).unsqueeze(0)
    attention_mask = torch.ones(1, inputs_embeds.shape[1], device=self.device,
                                dtype=torch.long)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_inference_loop.py::test_ephemeral_l2_reconstruction_included_in_context -v
```
Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/inference/inference_loop.py tests/test_inference_loop.py
git commit -m "feat: include ephemeral L2/L3 reconstructions in inference context"
```

---

## Verification

After both tasks, the conversation log behavior changes:

**Before:**
```
[CC] Miss      seg=ac3c5595  tier=l2  sim=0.7734  faults=1
[Inference] Reading L1=11 segs  native=11  promoted=0 (from L2:0 L3:0)
```

**After:**
```
[CC] Miss      seg=ac3c5595  tier=l2  sim=0.7734  faults=1
[Inference] Reading L1=11 segs  native=11  promoted=0 (from L2:0 L3:0)  ephemeral=1
```

The age segment is now included (lossily) in `inputs_embeds` even though it wasn't promoted.

Final sanity run:
```bash
python -m pytest tests/ -v --tb=short
```
All tests green = implementation complete.
