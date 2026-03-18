import uuid
import time
import contextlib
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

    fp = F.normalize(torch.ones(768), dim=0)
    sid = uuid.uuid4()
    anchors = SanityAnchors(["First.", "Last."], ["Alice"], torch.zeros(cfg.embed_dim))
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
            return contextlib.nullcontext()

    class MockTokenizer:
        eos_token_id = 0
        def encode(self, t, **kw): return [0] * 5
        def apply_chat_template(self, m, **kw): return torch.zeros(1, 3, dtype=torch.long)
        def decode(self, ids, **kw): return "ok"

    loop._llm = MockLLM()
    loop._tokenizer = MockTokenizer()
    loop.fingerprinter.encode = lambda text: fp

    loop.generate_response("How old am I?")

    assert "inputs_embeds" in captured, "generate() was never called"
    seq_len = captured["inputs_embeds"].shape[1]
    assert seq_len > 3, (
        f"Expected ephemeral L2 tokens to widen inputs_embeds beyond query length, got {seq_len}"
    )


def test_session_id_assigned(cfg):
    loop = InferenceLoop(cfg)
    assert loop.session_id is not None and len(loop.session_id) > 0


def test_process_text_increases_history(cfg):
    loop = InferenceLoop(cfg)
    loop.process_text("Hello world. This is a test sentence.")
    assert len(loop.conversation_history) > 0


def test_hit_rate_stats_structure(cfg):
    loop = InferenceLoop(cfg)
    stats = loop.hit_rate_stats
    for key in ("total_segments", "l1_count", "l2_count", "l3_count", "miss_events"):
        assert key in stats
