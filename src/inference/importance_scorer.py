from __future__ import annotations
import torch
from utils.config import Config

class ImportanceScorer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._mean = 0.0; self._var = 0.0; self._n = 0

    def score_from_attentions(
        self, attn: torch.Tensor, segment_start: int, segment_end: int
    ) -> float:
        """attn: [num_heads, seq_len, seq_len]. Mean attention subsequent tokens pay to each segment token."""
        token_scores = []
        for p_t in range(segment_start, segment_end):
            subsequent = list(range(p_t + 1, segment_end))
            if not subsequent:
                continue
            score = attn[:, subsequent, p_t].mean().item()
            token_scores.append(score)
        if not token_scores:
            return 0.0
        return self._z_score(sum(token_scores) / len(token_scores))

    def _z_score(self, value: float) -> float:
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        self._var += delta * (value - self._mean)
        std = (self._var / max(self._n - 1, 1)) ** 0.5
        return 0.0 if std < 1e-8 else (value - self._mean) / std

    def reset_session(self) -> None:
        self._mean = 0.0; self._var = 0.0; self._n = 0
