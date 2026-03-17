from inference_loop import InferenceLoop


def test_session_id_assigned(cfg):
    loop = InferenceLoop(cfg)
    assert loop.session_id is not None and len(loop.session_id) > 0


def test_process_text_increases_history(cfg):
    loop = InferenceLoop(cfg)
    loop.process_text("Hello world. This is a test sentence.")
    assert len(loop.conversation_history) > 0


def test_hit_rate_stats_structure(cfg):
    loop = InferenceLoop(cfg)
    stats = loop.hit_rate_stats
    for key in ("total_segments", "l1_count", "l2_count", "l3_count", "miss_events"):
        assert key in stats
