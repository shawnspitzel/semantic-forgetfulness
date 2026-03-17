import uuid, time
import torch
from memory.l1_store import L1Store
from utils.data_structures import L1Entry

def _entry(pos=0, importance=0.5, n=10, D=768):
    return L1Entry(
        id=uuid.uuid4(), tokens=torch.zeros(n, dtype=torch.long),
        token_embeddings=torch.zeros(n, D), importance_score=importance,
        last_accessed=time.time(), source_position=pos,
        session_id="s1", is_reconstructed=False,
    )

def test_insert_retrieve(cfg):
    store = L1Store(capacity=5, cfg=cfg)
    e = _entry()
    store.insert(e)
    assert store.get(e.id) is not None

def test_evicts_lowest_score(cfg):
    store = L1Store(capacity=2, cfg=cfg)
    low = _entry(pos=0, importance=0.1)
    high = _entry(pos=1, importance=0.9)
    store.insert(low); store.insert(high)
    evicted = store.insert(_entry(pos=2))
    assert evicted is not None and evicted.id == low.id

def test_ordered_by_position(cfg):
    store = L1Store(capacity=5, cfg=cfg)
    for pos in [3, 1, 2]:
        store.insert(_entry(pos=pos))
    positions = [e.source_position for e in store.get_all_by_position()]
    assert positions == sorted(positions)
