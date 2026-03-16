from __future__ import annotations
import time, uuid
from typing import Callable, Optional
import torch
import torch.nn.functional as F

from sf.config import Config
from sf.data_structures import (
    SegmentMetadata, SanityAnchors, MissEvent, L1Entry, L2Entry, L3Entry
)
from sf.entity_graph import EntityGraph
from sf.stores.l1 import L1Store
from sf.stores.l2 import L2Store
from sf.stores.l3 import L3Store


class CacheController:
    def __init__(
        self,
        cfg: Config,
        session_id: str,
        compress_fn: Optional[Callable] = None,
        reconstruct_fn: Optional[Callable] = None,
        embed_dim: Optional[int] = None,
    ):
        self.cfg = cfg
        self.session_id = session_id
        self.compress_fn = compress_fn
        self.reconstruct_fn = reconstruct_fn
        d = embed_dim or cfg.embed_dim

        self._entity_graph = EntityGraph()
        # L1 capacity: large enough to hold context before L2 activates
        l1_cap = max(cfg.l2_activation_threshold // max(cfg.segment_target, 1), 10)
        self._l1 = L1Store(capacity=l1_cap, cfg=cfg)
        self._l2 = L2Store(capacity=cfg.l2_capacity, cfg=cfg, entity_graph=self._entity_graph)
        self._l3 = L3Store(capacity=cfg.l3_capacity, cfg=cfg,
                           entity_graph=self._entity_graph, embed_dim=d)

        self._metadata: dict[uuid.UUID, SegmentMetadata] = {}
        self._anchors: dict[uuid.UUID, SanityAnchors] = {}
        self.miss_log: list[MissEvent] = []

    # ── Admission ────────────────────────────────────────────────────────

    def admit(
        self,
        segment_id: uuid.UUID,
        token_embeddings: torch.Tensor,
        importance_score: float,
        source_position: int,
        fingerprint: torch.Tensor,
        anchors: SanityAnchors,
        total_tokens_seen: int,
    ) -> None:
        meta = SegmentMetadata(
            id=segment_id, tier="l1", importance_score=importance_score,
            last_accessed=time.time(), source_position=source_position,
            session_id=self.session_id, is_reconstructed=False,
            semantic_fingerprint=fingerprint, fault_count=0,
            original_length=token_embeddings.shape[0],
        )
        self._metadata[segment_id] = meta
        self._anchors[segment_id] = anchors

        entry = L1Entry(
            id=segment_id,
            tokens=torch.zeros(token_embeddings.shape[0], dtype=torch.long),
            token_embeddings=token_embeddings,
            importance_score=importance_score,
            last_accessed=time.time(),
            source_position=source_position,
            session_id=self.session_id,
            is_reconstructed=False,
        )

        if total_tokens_seen < self.cfg.l2_activation_threshold:
            self._l1._entries[segment_id] = entry
            return

        evicted = self._l1.insert(entry)
        if evicted:
            self._demote_l1_to_l2(evicted, total_tokens_seen)

    def _demote_l1_to_l2(self, evicted: L1Entry, total_tokens_seen: int) -> None:
        if self.compress_fn is None:
            return
        anchors = self._anchors.get(evicted.id)
        ce = self.compress_fn(evicted.token_embeddings, self.cfg.C_L2)
        l2e = L2Entry(
            id=evicted.id, ce_tensor=ce,
            semantic_fingerprint=self._metadata[evicted.id].semantic_fingerprint,
            confidence_scores=None, importance_score=evicted.importance_score,
            last_accessed=time.time(), source_position=evicted.source_position,
            session_id=self.session_id, origin="l1_demotion",
            grounding_used=False, entities=anchors.entities if anchors else [],
        )
        self._metadata[evicted.id].tier = "l2"
        evicted_l2 = self._l2.insert(l2e)
        if evicted_l2 and total_tokens_seen >= self.cfg.l3_activation_threshold:
            self._demote_l2_to_l3(evicted_l2)

    def _demote_l2_to_l3(self, evicted: L2Entry) -> None:
        if self.compress_fn is None:
            return
        anchors = self._anchors.get(evicted.id)
        ce_l3 = self.compress_fn(evicted.ce_tensor, self.cfg.C_L3)
        l3e = L3Entry(
            id=evicted.id, concept_embeddings=ce_l3,
            representative_vec=ce_l3.mean(dim=0),
            sanity_anchors=anchors or SanityAnchors([], [], torch.zeros(768)),
            importance_score=evicted.importance_score, last_accessed=time.time(),
            source_position=evicted.source_position, session_id=self.session_id,
            l2_ce_at_demotion=evicted.ce_tensor,
            original_length=self._metadata[evicted.id].original_length,
        )
        self._metadata[evicted.id].tier = "l3"
        self._l3.insert(l3e)

    # ── Miss Detection ───────────────────────────────────────────────────

    def detect_misses(self, query_vec: torch.Tensor) -> list[MissEvent]:
        misses = []
        for seg_id, meta in self._metadata.items():
            if meta.tier == "l1":
                continue
            sim = F.cosine_similarity(
                query_vec.unsqueeze(0), meta.semantic_fingerprint.unsqueeze(0)
            ).clamp(-1.0, 1.0).item()
            if sim >= self.cfg.miss_detection_theta:
                miss_type = "soft" if meta.tier == "l2" else "hard"
                event = MissEvent(seg_id, miss_type, query_vec, time.time())
                misses.append(event)
                self.miss_log.append(event)
                meta.fault_count += 1
        return misses

    # ── Promotion ────────────────────────────────────────────────────────

    def promote_to_l1(self, segment_id: uuid.UUID, query_vec: torch.Tensor) -> Optional[L1Entry]:
        meta = self._metadata.get(segment_id)
        if meta is None or self.reconstruct_fn is None:
            return None
        anchors = self._anchors.get(segment_id)

        if meta.tier == "l2":
            l2e = self._l2.get(segment_id)
            if not l2e:
                return None
            neighbors = self._l2.get_neighbors(segment_id)
            result = self.reconstruct_fn(l2e.ce_tensor, anchors, neighbors, query_vec, "l2_to_l1")
        else:
            l3e = self._l3.get(segment_id)
            if not l3e:
                return None
            neighbors = self._l2.get_neighbors(segment_id)
            r_l3 = self.reconstruct_fn(l3e.concept_embeddings, anchors, neighbors, None, "l3_to_l2")
            if r_l3.fingerprint_sim < self.cfg.reconstruction_theta:
                return None
            result = self.reconstruct_fn(r_l3.ce_tensor, anchors, neighbors, query_vec, "l2_to_l1")

        if result.fingerprint_sim < self.cfg.reconstruction_theta:
            return None

        promoted = L1Entry(
            id=segment_id, tokens=torch.zeros(result.ce_tensor.shape[0], dtype=torch.long),
            token_embeddings=result.ce_tensor, importance_score=meta.importance_score,
            last_accessed=time.time(), source_position=meta.source_position,
            session_id=self.session_id, is_reconstructed=True,
        )
        meta.tier = "l1"; meta.fault_count = 0
        self._l1.insert(promoted)
        return promoted

    # ── Accessors ────────────────────────────────────────────────────────

    def get_metadata(self, segment_id: uuid.UUID) -> Optional[SegmentMetadata]:
        return self._metadata.get(segment_id)

    def update_importance(self, segment_id: uuid.UUID, score: float) -> None:
        m = self._metadata.get(segment_id)
        if m:
            m.importance_score = score

    def l1_entries(self) -> list[L1Entry]:
        return self._l1.get_all_by_position()

    @property
    def hit_rate_stats(self) -> dict:
        tiers = [m.tier for m in self._metadata.values()]
        total = len(tiers)
        return {
            "total_segments": total,
            "l1_count": tiers.count("l1"),
            "l2_count": tiers.count("l2"),
            "l3_count": tiers.count("l3"),
            "miss_events": len(self.miss_log),
        }
