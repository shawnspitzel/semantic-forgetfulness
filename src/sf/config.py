from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Base LLM — production default is Llama-3.2-3B-Instruct
    # Tests override to gpt2 via conftest.cfg fixture
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    embed_dim: int = 3072          # Llama-3.2-3B hidden_size; gpt2 uses 768

    # Segmentation
    segment_target: int = 20
    segment_hard_cap: int = 25

    # CE layout (must satisfy C_L2 > C_L3 > E + B)
    C_L2: int = 8                  # CE slots for L1->L2 (~2x compression from 20-token segment)
    C_L3: int = 5                  # CE slots for L2->L3 (~4x total from original)
    E: int = 2                     # entity anchor slots in CE layout
    B: int = 2                     # boundary slots (first + last sentence; fixed)

    # Cache sizes
    l2_capacity: int = 200
    l3_capacity: int = 1000

    # Tier activation (total tokens in session)
    l2_activation_threshold: int = 8_000
    l3_activation_threshold: int = 32_000

    # Eviction scoring
    alpha: float = 0.6             # importance weight
    lam: float = 0.1               # recency decay rate

    # Miss detection
    miss_detection_theta: float = 0.7
    leniency: int = 2              # fault_count before promotion

    # Reconstructor
    reconstruction_theta: float = 0.75
    tau: float = 0.7               # per-entity-slot cosine threshold
    reconstruction_retry_budget: int = 3

    # Importance scoring
    tsp_layer_index: int = -1      # -1 = floor(num_layers/2) at runtime

    # Training
    lora_rank: int = 16
    min_session_length: int = 20
    fine_tuning_steps: int = 1
    context_window_W: int = 100
    pretrain_learning_rate: float = 2e-4
    finetune_learning_rate: float = 5e-5

    def __post_init__(self):
        assert self.C_L2 > self.C_L3 > self.E + self.B, (
            f"CE layout constraint violated: C_L2({self.C_L2}) > "
            f"C_L3({self.C_L3}) > E+B({self.E + self.B})"
        )
