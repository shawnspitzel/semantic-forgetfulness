import pytest
import torch
from sf.config import Config

@pytest.fixture(scope="session")
def cfg():
    """Test config — uses gpt2 (small, no gated access) and CPU-friendly sizes."""
    return Config(
        model_name="gpt2",
        embed_dim=768,           # gpt2 hidden_size
        segment_target=10,
        segment_hard_cap=15,
        C_L2=8,
        C_L3=5,
        E=2,
        B=2,
        l2_capacity=10,
        l3_capacity=20,
        l2_activation_threshold=50,
        l3_activation_threshold=100,
    )

@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")
