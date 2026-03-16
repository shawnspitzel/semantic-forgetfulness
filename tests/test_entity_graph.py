import uuid
from sf.entity_graph import EntityGraph

def test_shared_entity_links_segments():
    g = EntityGraph()
    id1, id2 = uuid.uuid4(), uuid.uuid4()
    g.add_segment(id1, ["Paris", "Eiffel"])
    g.add_segment(id2, ["Paris", "Louvre"])
    assert id2 in g.get_segment_neighbors(id1)

def test_remove_cleans_up():
    g = EntityGraph()
    sid = uuid.uuid4()
    g.add_segment(sid, ["London"])
    g.remove_segment(sid)
    assert g.get_segment_neighbors(sid) == set()
