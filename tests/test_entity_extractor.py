from semantic.entity_extractor import EntityExtractor

def test_extracts_names():
    entities = EntityExtractor().extract("Barack Obama visited Paris.")
    assert len(entities) > 0

def test_empty_text():
    assert EntityExtractor().extract("") == []

def test_truncated_to_max(cfg):
    result = EntityExtractor(max_entities=cfg.E).extract(
        "Alice, Bob, Carol, and Dave met in London, Paris, and Tokyo."
    )
    assert len(result) <= cfg.E
