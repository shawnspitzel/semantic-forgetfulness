import torch
from sf.reconstructor import Reconstructor
from sf.data_structures import SanityAnchors, ReconstructionResult

def _anchors(cfg):
    fp = torch.nn.functional.normalize(torch.randn(768), dim=0)
    return SanityAnchors(["First.", "Last."], ["Alice"], fp)

def test_l3_to_l2_output_shape(cfg):
    r = Reconstructor(cfg, device="cpu")
    result = r.reconstruct(torch.randn(cfg.C_L3, cfg.embed_dim), _anchors(cfg), [], None, "l3_to_l2")
    assert isinstance(result, ReconstructionResult)
    assert result.ce_tensor.shape == (cfg.C_L2, cfg.embed_dim)

def test_l2_to_l1_output_shape(cfg):
    r = Reconstructor(cfg, device="cpu")
    result = r.reconstruct(torch.randn(cfg.C_L2, cfg.embed_dim), _anchors(cfg), [], torch.randn(cfg.embed_dim), "l2_to_l1")
    assert result.ce_tensor.shape[1] == cfg.embed_dim

def test_fingerprint_sim_in_range(cfg):
    r = Reconstructor(cfg, device="cpu")
    result = r.reconstruct(torch.randn(cfg.C_L2, cfg.embed_dim), _anchors(cfg), [], None, "l3_to_l2")
    assert -1.0 <= result.fingerprint_sim <= 1.0
