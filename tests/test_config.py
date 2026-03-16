from sf.config import Config

def test_default_model_is_llama():
    cfg = Config()
    assert "llama" in cfg.model_name.lower()

def test_ce_layout_constraint_passes():
    cfg = Config(C_L2=8, C_L3=5, E=2, B=2)
    assert cfg.C_L2 > cfg.C_L3 > cfg.E + cfg.B

def test_ce_layout_constraint_raises_on_bad_values():
    import pytest
    with pytest.raises(AssertionError):
        Config(C_L2=8, C_L3=5, E=4, B=2)  # E+B=6 > C_L3=5 — violates constraint

def test_standard_hyperparameter_defaults():
    cfg = Config()
    assert cfg.alpha == 0.6
    assert cfg.miss_detection_theta == 0.7
    assert cfg.reconstruction_theta == 0.75
    assert cfg.leniency == 2
