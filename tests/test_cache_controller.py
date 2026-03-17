import uuid, time
import torch
import torch.nn.functional as F
from memory.cache_controller import CacheController
from utils.data_structures import SanityAnchors, L2Entry, ReconstructionResult

def _anchors():
    return SanityAnchors(["First.", "Last."], ["Alice"], torch.randn(768))

def test_admission_assigns_l1(cfg):
    cc = CacheController(cfg=cfg, session_id="s1")
    sid = uuid.uuid4()
    cc.admit(sid, torch.randn(20, cfg.embed_dim), 0.5, 0,
             torch.randn(768), _anchors(), total_tokens_seen=0)
    assert cc.get_metadata(sid).tier == "l1"

def test_miss_detection_flags_l2_segment(cfg):
    cc = CacheController(cfg=cfg, session_id="s1")
    sid = uuid.uuid4()
    fp = F.normalize(torch.randn(768), dim=0)
    cc.admit(sid, torch.randn(20, cfg.embed_dim), 0.5, 0,
             fp, _anchors(), total_tokens_seen=0)
    cc._metadata[sid].tier = "l2"
    misses = cc.detect_misses(fp)
    assert any(m.segment_id == sid for m in misses)
    assert any(m.miss_type == "soft" for m in misses)

def test_fault_count_increments(cfg):
    cc = CacheController(cfg=cfg, session_id="s1")
    sid = uuid.uuid4()
    fp = F.normalize(torch.randn(768), dim=0)
    cc.admit(sid, torch.randn(20, cfg.embed_dim), 0.5, 0,
             fp, _anchors(), total_tokens_seen=0)
    cc._metadata[sid].tier = "l2"
    cc.detect_misses(fp)
    assert cc._metadata[sid].fault_count == 1


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
