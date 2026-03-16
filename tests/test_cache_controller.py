import uuid, time
import torch
import torch.nn.functional as F
from sf.cache_controller import CacheController
from sf.data_structures import SanityAnchors

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
