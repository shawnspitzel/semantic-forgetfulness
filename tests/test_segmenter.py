from inference.segmenter import Segmenter

def test_hard_cap_respected(cfg):
    seg = Segmenter(cfg)
    tokens = list(range(100))
    for s in seg.segment(tokens):
        assert len(s) <= cfg.segment_hard_cap

def test_partial_flush(cfg):
    seg = Segmenter(cfg)
    assert seg.segment(list(range(5))) == [list(range(5))]

def test_empty(cfg):
    assert Segmenter(cfg).segment([]) == []
