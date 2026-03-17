from __future__ import annotations
import math, time, uuid
from config import Config
from data_structures import L1Entry

class L1Store:
    def __init__(self, capacity: int, cfg: Config):
        self.capacity = capacity
        self.cfg = cfg
        self._entries: dict[uuid.UUID, L1Entry] = {}

    def insert(self, entry: L1Entry) -> L1Entry | None:
        evicted = None
        if len(self._entries) >= self.capacity:
            evicted = self._evict()
        self._entries[entry.id] = entry
        return evicted

    def get(self, segment_id: uuid.UUID) -> L1Entry | None:
        e = self._entries.get(segment_id)
        if e:
            e.last_accessed = time.time()
        return e

    def remove(self, segment_id: uuid.UUID) -> L1Entry | None:
        return self._entries.pop(segment_id, None)

    def get_all_by_position(self) -> list[L1Entry]:
        return sorted(self._entries.values(), key=lambda e: e.source_position)

    def __len__(self) -> int:
        return len(self._entries)

    def _score(self, e: L1Entry) -> float:
        rec = math.exp(-self.cfg.lam * (time.time() - e.last_accessed))
        return self.cfg.alpha * e.importance_score + (1 - self.cfg.alpha) * rec

    def _evict(self) -> L1Entry:
        tid = min(self._entries, key=lambda eid: self._score(self._entries[eid]))
        return self._entries.pop(tid)
