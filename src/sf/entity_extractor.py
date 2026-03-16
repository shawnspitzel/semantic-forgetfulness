from __future__ import annotations
import spacy

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _nlp

class EntityExtractor:
    def __init__(self, max_entities: int = 4):
        self.max_entities = max_entities

    def extract(self, text: str) -> list[str]:
        if not text.strip():
            return []
        seen: list[str] = []
        for ent in _get_nlp()(text).ents:
            label = ent.text.strip()
            if label and label not in seen:
                seen.append(label)
                if len(seen) >= self.max_entities:
                    break
        return seen
