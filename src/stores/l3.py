from __future__ import annotations
import math, time, uuid
import numpy as np
import torch
from usearch.index import Index
from config import Config
from data_structures import L3Entry
from entity_graph import EntityGraph

class L3Store:
    def __init__(self, capacity: int, cfg: Config, entity_graph: EntityGraph, embed_dim: int):
        self.capacity = capacity
        self.cfg = cfg
        self.graph = entity_graph
        self.embed_dim = embed_dim
        self._entries: dict[uuid.UUID, L3Entry] = {}
        self._index = Index(ndim=embed_dim, metric='cos')
        self._uuid_to_label: dict[uuid.UUID, int] = {}
        self._label_to_uuid: dict[int, uuid.UUID] = {}
        self._evicted_labels: set[int] = set()   # labels removed but still in index
        self._next_label = 0

    def insert(self, entry: L3Entry) -> L3Entry | None:
        dropped = None
        if len(self._entries) >= self.capacity:
            dropped = self._evict()
        label = self._next_label; self._next_label += 1
        self._entries[entry.id] = entry
        self._uuid_to_label[entry.id] = label
        self._label_to_uuid[label] = entry.id
        vec = entry.representative_vec.float().cpu().numpy().astype(np.float32)
        self._index.add(label, vec)
        self.graph.add_segment(entry.id, entry.sanity_anchors.entities)
        return dropped

    def get(self, segment_id: uuid.UUID) -> L3Entry | None:
        e = self._entries.get(segment_id)
        if e:
            e.last_accessed = time.time()
        return e

    def search(self, query_vec: torch.Tensor, top_k: int) -> list[L3Entry]:
        if not self._entries:
            return []
        k = min(top_k, len(self._entries))
        vec = query_vec.float().cpu().numpy().astype(np.float32)
        matches = self._index.search(vec, count=k)
        seen, results = set(), []
        for label in matches.keys:
            label = int(label)
            if label in self._evicted_labels:
                continue
            uid = self._label_to_uuid.get(label)
            if uid and uid in self._entries:
                results.append(self._entries[uid])
                seen.add(uid)
        # 1-hop entity expansion
        for r in list(results):
            for nid in self.graph.get_segment_neighbors(r.id):
                if nid in self._entries and nid not in seen:
                    results.append(self._entries[nid])
                    seen.add(nid)
        return results

    def remove(self, segment_id: uuid.UUID) -> L3Entry | None:
        e = self._entries.pop(segment_id, None)
        if e:
            self.graph.remove_segment(segment_id)
            label = self._uuid_to_label.pop(segment_id, None)
            if label is not None:
                self._label_to_uuid.pop(label, None)
                self._evicted_labels.add(label)  # can't delete from usearch; mark as dead
        return e

    def __len__(self) -> int:
        return len(self._entries)

    def _score(self, e: L3Entry) -> float:
        rec = math.exp(-self.cfg.lam * (time.time() - e.last_accessed))
        return self.cfg.alpha * e.importance_score + (1 - self.cfg.alpha) * rec

    def _evict(self) -> L3Entry:
        tid = min(self._entries, key=lambda eid: self._score(self._entries[eid]))
        return self.remove(tid)
