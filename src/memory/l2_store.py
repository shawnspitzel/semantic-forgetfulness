from __future__ import annotations
import math, time, uuid
import torch
import torch.nn.functional as F
from utils.config import Config
from utils.data_structures import L2Entry
from semantic.entity_graph import EntityGraph

class L2Store:
    def __init__(self, capacity: int, cfg: Config, entity_graph: EntityGraph):
        self.capacity = capacity
        self.cfg = cfg
        self.graph = entity_graph
        self._entries: dict[uuid.UUID, L2Entry] = {}

    def insert(self, entry: L2Entry) -> L2Entry | None:
        evicted = None
        if len(self._entries) >= self.capacity:
            evicted = self._evict()
        self._entries[entry.id] = entry
        self.graph.add_segment(entry.id, entry.entities)
        return evicted

    def get(self, segment_id: uuid.UUID) -> L2Entry | None:
        e = self._entries.get(segment_id)
        if e:
            e.last_accessed = time.time()
        return e

    def remove(self, segment_id: uuid.UUID) -> L2Entry | None:
        e = self._entries.pop(segment_id, None)
        if e:
            self.graph.remove_segment(segment_id)
        return e

    def search_by_fingerprint(self, query_vec: torch.Tensor, top_k: int, theta: float) -> list[L2Entry]:
        if not self._entries:
            return []
        ids = list(self._entries.keys())
        fps = torch.stack([self._entries[i].semantic_fingerprint for i in ids])
        sims = F.cosine_similarity(query_vec.unsqueeze(0), fps).clamp(-1.0, 1.0)
        passing = [(ids[i], sims[i].item()) for i in range(len(ids)) if sims[i].item() >= theta]
        if not passing:
            return []
        passing.sort(key=lambda x: x[1], reverse=True)
        return [self._entries[uid] for uid, _ in passing[:top_k]]

    def get_neighbors(self, segment_id: uuid.UUID) -> list[L2Entry]:
        return [self._entries[nid] for nid in self.graph.get_segment_neighbors(segment_id)
                if nid in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def _score(self, e: L2Entry) -> float:
        rec = math.exp(-self.cfg.lam * (time.time() - e.last_accessed))
        return self.cfg.alpha * e.importance_score + (1 - self.cfg.alpha) * rec

    def _evict(self) -> L2Entry:
        tid = min(self._entries, key=lambda eid: self._score(self._entries[eid]))
        return self.remove(tid)
