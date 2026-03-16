from pathlib import Path
import pytest
from sf.config import Config

def test_load_from_yaml():
    """Config.load() reads config.yaml from repo root."""
    cfg = Config.load()
    assert "llama" in cfg.model_name.lower()
    assert cfg.C_L2 > cfg.C_L3 > cfg.E + cfg.B

def test_direct_construction_still_works():
    """Config can still be constructed directly (used in tests/conftest.py)."""
    cfg = Config(
        model_name="gpt2", embed_dim=768, segment_target=10, segment_hard_cap=15,
        C_L2=8, C_L3=5, E=2, B=2, l2_capacity=10, l3_capacity=20,
        l2_activation_threshold=50, l3_activation_threshold=100,
        alpha=0.6, lam=0.1, miss_detection_theta=0.7, leniency=2,
        reconstruction_theta=0.75, tau=0.7, reconstruction_retry_budget=3,
        tsp_layer_index=-1, lora_rank=16, min_session_length=20,
        fine_tuning_steps=1, context_window_W=100,
        pretrain_learning_rate=2e-4, finetune_learning_rate=5e-5,
    )
    assert cfg.model_name == "gpt2"

def test_ce_layout_constraint_passes():
    cfg = Config.load()
    assert cfg.C_L2 > cfg.C_L3 > cfg.E + cfg.B

def test_ce_layout_constraint_raises_on_bad_values():
    with pytest.raises(AssertionError):
        Config(
            model_name="gpt2", embed_dim=768, segment_target=10, segment_hard_cap=15,
            C_L2=8, C_L3=5, E=4, B=2,  # E+B=6 > C_L3=5 — violates constraint
            l2_capacity=10, l3_capacity=20, l2_activation_threshold=50,
            l3_activation_threshold=100, alpha=0.6, lam=0.1,
            miss_detection_theta=0.7, leniency=2, reconstruction_theta=0.75,
            tau=0.7, reconstruction_retry_budget=3, tsp_layer_index=-1,
            lora_rank=16, min_session_length=20, fine_tuning_steps=1,
            context_window_W=100, pretrain_learning_rate=2e-4, finetune_learning_rate=5e-5,
        )

def test_standard_hyperparameter_defaults():
    cfg = Config.load()
    assert cfg.alpha == 0.6
    assert cfg.miss_detection_theta == 0.7
    assert cfg.reconstruction_theta == 0.75
    assert cfg.leniency == 2
