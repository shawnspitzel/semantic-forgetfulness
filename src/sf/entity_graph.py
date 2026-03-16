from __future__ import annotations
import uuid
from collections import defaultdict
import networkx as nx

class EntityGraph:
    def __init__(self):
        self._graph = nx.Graph()
        self._seg_to_entities: dict[uuid.UUID, set[str]] = defaultdict(set)
        self._entity_to_segs: dict[str, set[uuid.UUID]] = defaultdict(set)

    def add_segment(self, segment_id: uuid.UUID, entities: list[str]) -> None:
        for e in entities:
            self._seg_to_entities[segment_id].add(e)
            self._entity_to_segs[e].add(segment_id)
            if not self._graph.has_node(e):
                self._graph.add_node(e)
        ents = list(entities)
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                if not self._graph.has_edge(ents[i], ents[j]):
                    self._graph.add_edge(ents[i], ents[j], weight=1)
                else:
                    self._graph[ents[i]][ents[j]]["weight"] += 1

    def remove_segment(self, segment_id: uuid.UUID) -> None:
        for e in self._seg_to_entities.pop(segment_id, set()):
            self._entity_to_segs[e].discard(segment_id)

    def get_segment_neighbors(self, segment_id: uuid.UUID) -> set[uuid.UUID]:
        result: set[uuid.UUID] = set()
        for e in self._seg_to_entities.get(segment_id, set()):
            result.update(self._entity_to_segs.get(e, set()))
        result.discard(segment_id)
        return result

    def get_neighbors_for_entities(self, entities: list[str]) -> set[uuid.UUID]:
        result: set[uuid.UUID] = set()
        for e in entities:
            result.update(self._entity_to_segs.get(e, set()))
        return result
