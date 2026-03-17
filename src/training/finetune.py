"""Phase 2: End-of-session LoRA fine-tuning from cache-miss log."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from utils.config import Config
from utils.data_structures import FullMissEvent
from compression.compressor import Compressor
from compression.reconstructor import Reconstructor


def run_finetune(
    cfg: Config,
    miss_events: list[FullMissEvent],
    compressor: Compressor,
    reconstructor: Reconstructor,
    device: torch.device,
) -> dict:
    """Single gradient step over missed segments. No-op below min_session_length."""
    if len(miss_events) < cfg.min_session_length:
        return {"skipped": True, "reason": "insufficient_miss_events",
                "count": len(miss_events)}

    params = list(compressor.parameters()) + list(reconstructor.parameters())
    optimizer = AdamW(params, lr=cfg.finetune_learning_rate)
    total_loss = 0.0

    for _ in range(cfg.fine_tuning_steps):
        optimizer.zero_grad()
        step_loss = torch.tensor(0.0, device=device)

        for event in miss_events:
            inp = event.segment_input.to(device)
            target = inp.mean(dim=0)
            target_c = cfg.C_L2 if event.miss_level == "l2" else cfg.C_L3
            ce = compressor.compress(inp, target_c)
            loss = 1.0 - F.cosine_similarity(
                ce.mean(dim=0).unsqueeze(0), target.unsqueeze(0)
            ).mean()
            step_loss = step_loss + loss

        step_loss = step_loss / max(len(miss_events), 1)
        step_loss.backward()
        optimizer.step()
        total_loss += step_loss.item()

    return {
        "skipped": False,
        "miss_events_used": len(miss_events),
        "avg_loss": total_loss / cfg.fine_tuning_steps,
    }
