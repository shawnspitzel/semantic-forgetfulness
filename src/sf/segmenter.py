from __future__ import annotations
from sf.config import Config

_SENTENCE_END_CHARS = frozenset([".", "?", "!", "\n"])

class Segmenter:
    def __init__(self, cfg: Config, boundary_token_ids: set[int] | None = None):
        self.cfg = cfg
        self.boundary_ids = boundary_token_ids or set()

    def segment(self, token_ids: list[int]) -> list[list[int]]:
        if not token_ids:
            return []
        segments, buf = [], []
        for tid in token_ids:
            buf.append(tid)
            at_boundary = tid in self.boundary_ids
            if len(buf) >= self.cfg.segment_hard_cap or (
                len(buf) >= self.cfg.segment_target and at_boundary
            ):
                segments.append(buf)
                buf = []
        if buf:
            segments.append(buf)
        return segments

    def segment_text(self, text: str, tokenizer) -> list[list[int]]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        self.boundary_ids = {
            tokenizer.encode(ch, add_special_tokens=False)[0]
            for ch in _SENTENCE_END_CHARS
            if tokenizer.encode(ch, add_special_tokens=False)
        }
        return self.segment(ids)
