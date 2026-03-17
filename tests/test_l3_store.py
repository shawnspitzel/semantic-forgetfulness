import uuid, time
import torch
from memory.l3_store import L3Store
from utils.data_structures import L3Entry, SanityAnchors
from semantic.entity_graph import EntityGraph

def _entry(D=768, C3=5, C2=8):
    return L3Entry(
        id=uuid.uuid4(),
        concept_embeddings=torch.randn(C3, D),
        representative_vec=torch.nn.functional.normalize(torch.randn(D), dim=0),
        sanity_anchors=SanityAnchors(["A.", "B."], ["X"], torch.randn(768)),
        importance_score=0.5, last_accessed=time.time(),
        source_position=0, session_id="s1",
        l2_ce_at_demotion=torch.randn(C2, D), original_length=20,
    )

def test_insert_and_search(cfg):
    store = L3Store(10, cfg, EntityGraph(), embed_dim=768)
    e = _entry()
    store.insert(e)
    results = store.search(e.representative_vec, top_k=1)
    assert len(results) >= 1

def test_eviction_drops_permanently(cfg):
    store = L3Store(2, cfg, EntityGraph(), embed_dim=768)
    low = _entry(); low.importance_score = 0.1
    high = _entry(); high.importance_score = 0.9
    store.insert(low); store.insert(high)
    dropped = store.insert(_entry())
    assert dropped is not None
    assert store.get(dropped.id) is None        # evicted entry is gone
    assert len(store) == 2                       # capacity respected
