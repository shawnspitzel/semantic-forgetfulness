from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Config:
    # Base LLM
    model_name: str
    embed_dim: int

    # Segmentation
    segment_target: int
    segment_hard_cap: int

    # CE layout
    C_L2: int
    C_L3: int
    E: int
    B: int

    # Cache sizes
    l2_capacity: int
    l3_capacity: int

    # Tier activation
    l2_activation_threshold: int
    l3_activation_threshold: int

    # Eviction scoring
    alpha: float
    lam: float

    # Miss detection
    miss_detection_theta: float
    leniency: int

    # Reconstructor
    reconstruction_theta: float
    tau: float
    reconstruction_retry_budget: int

    # Importance scoring
    tsp_layer_index: int

    # Memory layer
    memory_enabled: bool

    # Training
    lora_rank: int
    min_session_length: int
    fine_tuning_steps: int
    context_window_W: int
    pretrain_learning_rate: float
    finetune_learning_rate: float

    def __post_init__(self):
        assert self.C_L2 > self.C_L3 > self.E + self.B, (
            f"CE layout constraint violated: C_L2({self.C_L2}) > "
            f"C_L3({self.C_L3}) > E+B({self.E + self.B})"
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load config from YAML file. Defaults to config.yaml at repo root."""
        if path is None:
            # Walk up from this file to find config.yaml at repo root
            here = Path(__file__).resolve().parent
            # src/sf/config.py -> walk up to find config.yaml
            for candidate in [here.parent.parent / "config.yaml",
                               here.parent / "config.yaml",
                               Path("config.yaml")]:
                if candidate.exists():
                    path = candidate
                    break
            else:
                raise FileNotFoundError(
                    "config.yaml not found. Expected at repo root or src/"
                )
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)
