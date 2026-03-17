import torch
from inference.importance_scorer import ImportanceScorer

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

def test_z_score_first_call_returns_zero(cfg):
    """Single observation has no variance — z-score should be 0."""
    scorer = ImportanceScorer(cfg)
    # With only one data point, std ≈ 0, so _z_score returns 0.0
    result = scorer._z_score(5.0)
    assert result == 0.0  # std < 1e-8 when n=1, var=0
