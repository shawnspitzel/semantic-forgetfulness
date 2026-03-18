import uuid, time
import torch
from utils.data_structures import SanityAnchors, SegmentMetadata, L1Entry, L2Entry, L3Entry

# Must match embed_dim in conftest cfg (gpt2 = 768).
D = 768

def test_l3entry_has_l2_ce_at_demotion():
    """inference.md I5: L3Entry must store l2_ce_at_demotion for miss reconciliation."""
    entry = L3Entry(
        id=uuid.uuid4(),
        concept_embeddings=torch.zeros(5, 768),
        representative_vec=torch.zeros(768),
        sanity_anchors=SanityAnchors([], [], torch.zeros(D)),
        importance_score=0.5,
        last_accessed=time.time(),
        source_position=0,
        session_id="s1",
        l2_ce_at_demotion=torch.zeros(8, 768),
        original_length=20,
    )
    assert entry.l2_ce_at_demotion.shape == (8, 768)

def test_segment_metadata_fields():
    meta = SegmentMetadata(
        id=uuid.uuid4(), tier="l1", importance_score=0.5,
        last_accessed=time.time(), source_position=0, session_id="s1",
        is_reconstructed=False, semantic_fingerprint=torch.zeros(768),
        fault_count=0, original_length=20,
    )
    assert meta.tier == "l1"
    assert meta.fault_count == 0
