import torch
from sf.importance_scorer import ImportanceScorer

def test_returns_float(cfg):
    scorer = ImportanceScorer(cfg)
    attn = torch.rand(12, 20, 20)
    score = scorer.score_from_attentions(attn, segment_start=10, segment_end=20)
    assert isinstance(score, float)

def test_high_attention_scores_higher(cfg):
    scorer = ImportanceScorer(cfg)
    seq_len = 20
    high = torch.zeros(1, seq_len, seq_len)
    for i in range(10, seq_len):
        high[0, i, 10:i] = 1.0
    low = torch.zeros(1, seq_len, seq_len)
    assert scorer.score_from_attentions(high, 10, 20) >= scorer.score_from_attentions(low, 10, 20)
