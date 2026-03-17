import torch
from compressor import Compressor

def test_l1_to_l2_shape(cfg):
    comp = Compressor(cfg, device="cpu")
    ce = comp.compress(torch.randn(20, cfg.embed_dim), target_c=cfg.C_L2)
    assert ce.shape == (cfg.C_L2, cfg.embed_dim)

def test_l2_to_l3_shape(cfg):
    comp = Compressor(cfg, device="cpu")
    ce = comp.compress(torch.randn(cfg.C_L2, cfg.embed_dim), target_c=cfg.C_L3)
    assert ce.shape == (cfg.C_L3, cfg.embed_dim)

def test_lora_params_require_grad(cfg):
    comp = Compressor(cfg, device="cpu")
    trainable = [p for p in comp.parameters()]
    assert len(trainable) > 0
    assert all(p.requires_grad for p in trainable)
