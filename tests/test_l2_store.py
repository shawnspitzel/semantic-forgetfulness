import uuid, time
import torch
import torch.nn.functional as F
from sf.stores.l2 import L2Store
from sf.data_structures import L2Entry
from sf.entity_graph import EntityGraph

def _entry(pos=0, importance=0.5, C=8, D=768):
    return L2Entry(
        id=uuid.uuid4(), ce_tensor=torch.randn(C, D),
        semantic_fingerprint=torch.randn(768),
        confidence_scores=None, importance_score=importance,
        last_accessed=time.time(), source_position=pos,
        session_id="s1", origin="l1_demotion",
        grounding_used=False, entities=["Alice"],
    )

def test_insert_get(cfg):
    store = L2Store(5, cfg, EntityGraph())
    e = _entry()
    store.insert(e)
    assert store.get(e.id) is not None

def test_evicts_lowest(cfg):
    store = L2Store(2, cfg, EntityGraph())
    low = _entry(importance=0.1); high = _entry(importance=0.9)
    store.insert(low); store.insert(high)
    evicted = store.insert(_entry())
    assert evicted is not None and evicted.id == low.id

def test_fingerprint_search(cfg):
    store = L2Store(10, cfg, EntityGraph())
    target = F.normalize(torch.randn(768), dim=0)
    e = _entry(); e.semantic_fingerprint = target
    store.insert(e)
    results = store.search_by_fingerprint(target, top_k=1, theta=0.0)
    assert results[0].id == e.id
