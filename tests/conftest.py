import pytest
import torch
from utils.config import Config

@pytest.fixture(scope="session")
def cfg():
    """Test config — uses gpt2 (small, no gated access) and CPU-friendly sizes."""
    return Config(
        model_name="gpt2",
        embed_dim=768,
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
        # Remaining fields with test-appropriate values matching config.yaml defaults:
        alpha=0.6,
        lam=0.1,
        miss_detection_theta=0.7,
        leniency=2,
        reconstruction_theta=0.75,
        tau=0.7,
        reconstruction_retry_budget=3,
        tsp_layer_index=-1,
        lora_rank=16,
        min_session_length=20,
        fine_tuning_steps=1,
        context_window_W=100,
        pretrain_learning_rate=2e-4,
        finetune_learning_rate=5e-5,
    )

@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")
