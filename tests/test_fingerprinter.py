import torch
from semantic.fingerprinter import Fingerprinter

def test_shape():
    assert Fingerprinter().encode("hello").shape == (384,)

def test_deterministic():
    fp = Fingerprinter()
    assert torch.allclose(fp.encode("test"), fp.encode("test"))

def test_cosine_self_is_one():
    fp = Fingerprinter()
    v = fp.encode("Transformers are neural networks.")
    sim = torch.nn.functional.cosine_similarity(v.unsqueeze(0), v.unsqueeze(0))
    assert sim.item() > 0.999
