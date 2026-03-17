from __future__ import annotations
import uuid, time
from dataclasses import dataclass
from typing import Literal, Optional
import torch

Tier = Literal["l1", "l2", "l3"]
MissType = Literal["soft", "hard", "total"]

@dataclass
class SanityAnchors:
    boundary_sentences: list[str]
    entities: list[str]
    semantic_fingerprint: torch.Tensor     # [768]

@dataclass
class SegmentMetadata:
    id: uuid.UUID
    tier: Tier
    importance_score: float
    last_accessed: float
    source_position: int
    session_id: str
    is_reconstructed: bool
    semantic_fingerprint: torch.Tensor     # [768]
    fault_count: int
    original_length: int

@dataclass
class L1Entry:
    id: uuid.UUID
    tokens: torch.Tensor                   # [T] token ids
    token_embeddings: torch.Tensor         # [T, D]
    importance_score: float
    last_accessed: float
    source_position: int
    session_id: str
    is_reconstructed: bool

@dataclass
class L2Entry:
    id: uuid.UUID
    ce_tensor: torch.Tensor                # [C_L2, D]
    semantic_fingerprint: torch.Tensor     # [768]
    confidence_scores: Optional[list[tuple[int, float]]]
    importance_score: float
    last_accessed: float
    source_position: int
    session_id: str
    origin: Literal["l1_demotion", "l3_promotion"]
    grounding_used: bool
    entities: list[str]

@dataclass
class L3Entry:
    id: uuid.UUID
    concept_embeddings: torch.Tensor       # [C_L3, D]
    representative_vec: torch.Tensor       # [D] mean-pool for ANN lookup
    sanity_anchors: SanityAnchors
    importance_score: float
    last_accessed: float
    source_position: int
    session_id: str
    l2_ce_at_demotion: torch.Tensor        # [C_L2, D] required for miss reconciliation
    original_length: int

@dataclass
class MissEvent:
    segment_id: uuid.UUID
    miss_type: MissType
    query_vec: torch.Tensor                # [768]
    timestamp: float

@dataclass
class FullMissEvent:
    """Reconciled at session end for fine-tuning."""
    segment_input: torch.Tensor            # [T, D] for L2 (token embeddings) or [C_L2, D] for L3 miss
    context_window: torch.Tensor           # [W, D]
    ce_produced: torch.Tensor              # [C_L2, D] or [C_L3, D] depending on miss_level
    miss_level: Literal["l2", "l3"]
    session_position: int

@dataclass
class ReconstructionResult:
    ce_tensor: torch.Tensor
    confidence_scores: list[tuple[int, float]]
    fingerprint_sim: float
    grounding_used: bool
    fallback: bool
