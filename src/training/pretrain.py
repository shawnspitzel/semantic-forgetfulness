"""
Phase 1: Distillation pretraining.

Trains compressor+reconstructor so that CEs are valid soft-token substitutes
for the frozen LLM backbone.

Usage:
  python -m sf.training.pretrain --data-path data/train.txt --steps 500 --device cuda
  python -m sf.training.pretrain --data-path data/train.txt --steps 500 --device cuda --wandb
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.config import Config
from compression.compressor import Compressor
from compression.reconstructor import Reconstructor
from inference.segmenter import Segmenter
from semantic.fingerprinter import Fingerprinter
from semantic.entity_extractor import EntityExtractor
from utils.data_structures import SanityAnchors
from utils.profiler import TrainingProfiler


def _grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return total ** 0.5


_CHUNK_CHARS = 10 * 1024 * 1024  # 10 MB per tokenization call


def _stream_segments(data_path: Path, tokenizer, cfg: Config):
    """
    Yield segments one at a time by streaming the file in 10 MB text chunks.

    Avoids two memory traps in the original code:
      1. read_text() loads the entire 2 GB file into a Python string.
      2. tokenizer.encode(full_text) produces a Python list[int] where each int
         is a 28-byte heap object — ~14 GB for a 2 GB corpus.

    At any point only one 10 MB chunk of text and a carry-over buf of at most
    segment_hard_cap token IDs live in memory simultaneously.
    """
    segmenter = Segmenter(cfg)
    buf: list[int] = []

    with data_path.open(encoding="utf-8") as f:
        while True:
            chunk = f.read(_CHUNK_CHARS)
            if not chunk:
                break
            buf.extend(tokenizer.encode(chunk, add_special_tokens=False))
            new_segs = segmenter.segment(buf)
            if len(new_segs) > 1:
                yield from new_segs[:-1]
                buf = new_segs[-1]   # carry the potentially-incomplete tail

    if buf:
        yield from segmenter.segment(buf)


def train(cfg: Config, data_path: Path, steps: int, device: str = "cpu",
          use_wandb: bool = False, profile_steps: int = 0) -> None:
    dev = torch.device(device)

    if profile_steps > 0 and steps < profile_steps + 3:
        raise ValueError(
            f"--profile {profile_steps} requires at least {profile_steps + 3} "
            f"total steps, but --steps {steps} was given."
        )

    if use_wandb:
        import wandb
        wandb.init(
            project="semantic-forgetfulness",
            config={
                "model": cfg.model_name,
                "lora_rank": cfg.lora_rank,
                "C_L2": cfg.C_L2,
                "C_L3": cfg.C_L3,
                "E": cfg.E,
                "B": cfg.B,
                "lr": cfg.pretrain_learning_rate,
                "steps": steps,
                "device": device,
            },
        )

    try:
        profiler = TrainingProfiler(use_wandb=use_wandb, profile_steps=profile_steps)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.bfloat16
        ).to(dev)
        for p in llm.parameters():
            p.requires_grad_(False)

        compressor = Compressor(cfg, device=dev)
        reconstructor = Reconstructor(cfg, device=dev)
        fingerprinter = Fingerprinter(str(dev))
        reconstructor.set_fingerprinter(fingerprinter)
        extractor = EntityExtractor(max_entities=cfg.E)

        params = list(compressor.parameters()) + list(reconstructor.parameters())
        optimizer = AdamW(params, lr=cfg.pretrain_learning_rate)

        num_layers = llm.config.num_hidden_layers
        layer_range = list(range(num_layers // 2, num_layers))
        embed = llm.get_input_embeddings()

        step = 0
        while step < steps:
            for seg_ids in _stream_segments(data_path, tokenizer, cfg):
                if step >= steps:
                    break

                with profiler.phase("data_load"):
                    seg_tensor = torch.tensor([seg_ids], device=dev)

                with profiler.phase("llm_forward_full"):
                    with torch.no_grad():
                        orig_embeds = embed(seg_tensor)[0]                 # [N, D]
                        out_full = llm(seg_tensor, output_hidden_states=True)
                        h_full_layers = {
                            layer_idx: out_full.hidden_states[layer_idx][:, -1, :].detach()
                            for layer_idx in layer_range
                        }
                        del out_full   # free all 28 layers of hidden states immediately

                with profiler.phase("entity_extraction"):
                    seg_text = tokenizer.decode(seg_ids)
                    entities = extractor.extract(seg_text)
                    sents = [s.strip() for s in seg_text.split(".") if s.strip()]
                    anchors = SanityAnchors(
                        boundary_sentences=[sents[0] if sents else "", sents[-1] if sents else ""],
                        entities=entities, semantic_fingerprint=orig_embeds.mean(dim=0).detach().cpu(),
                    )

                with profiler.phase("compressor"):
                    ce_l2 = compressor.compress(orig_embeds, cfg.C_L2)   # [C_L2, D]
                    ce_seq = ce_l2.unsqueeze(0)                           # [1, C_L2, D]

                with profiler.phase("llm_forward_ce"):
                    # L_distill: match hidden states at mid-to-late layers
                    out_ce = llm(inputs_embeds=ce_seq, output_hidden_states=True)
                    l_distill = torch.tensor(0.0, device=dev)
                    layer_losses: dict[str, float] = {}
                    for layer_idx in layer_range:
                        h_full = h_full_layers[layer_idx]
                        h_ce = out_ce.hidden_states[layer_idx][:, -1, :]
                        std = h_full.std().clamp(min=1e-6)
                        ll = F.smooth_l1_loss(h_ce / std, h_full / std)
                        l_distill = l_distill + ll
                        layer_losses[f"l_distill/layer_{layer_idx}"] = ll.item()
                    l_distill = l_distill / max(len(layer_range), 1)
                    del out_ce   # free all 28 layers of hidden states immediately

                with profiler.phase("reconstructor"):
                    # L_recon: reconstruct -> match original embedding mean
                    result = reconstructor.reconstruct(ce_l2, anchors, [], None, "l3_to_l2")
                    recon_mean = result.ce_tensor.mean(dim=0)
                    orig_mean = orig_embeds.mean(dim=0)
                    l_recon = 1.0 - F.cosine_similarity(recon_mean.unsqueeze(0), orig_mean.unsqueeze(0)).mean()

                loss = l_distill + l_recon

                with profiler.phase("backward"):
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

                with profiler.phase("optimizer"):
                    optimizer.step()

                step += 1

                if step % 10 == 0:
                    slot_norms = ce_l2.norm(dim=-1).detach()
                    confidence_scores = [s for _, s in result.confidence_scores] if result.confidence_scores else [0.0]
                    metrics = {
                        "loss/total": loss.item(),
                        "loss/l_distill": l_distill.item(),
                        "loss/l_recon": l_recon.item(),
                        "grad_norm/compressor": _grad_norm(compressor),
                        "grad_norm/reconstructor": _grad_norm(reconstructor),
                        "ce/slot_norm_mean": slot_norms.mean().item(),
                        "ce/slot_norm_std": slot_norms.std().item(),
                        "reconstruction/fingerprint_sim": result.fingerprint_sim if result.fingerprint_sim is not None else 0.0,
                        "reconstruction/confidence_mean": sum(confidence_scores) / len(confidence_scores),
                        "reconstruction/confidence_min": min(confidence_scores),
                        "reconstruction/fallback": float(result.fallback),
                        "reconstruction/grounding_used": float(result.grounding_used),
                        **layer_losses,
                    }
                else:
                    metrics = {}

                profiler.step()

                profiler.log_step(step, metrics, n_tokens=len(seg_ids), n_segments=1)

                if step % 50 == 0:
                    print(f"Step {step}/{steps}  L_distill={l_distill.item():.4f}  L_recon={l_recon.item():.4f}")

        Path("checkpoints").mkdir(exist_ok=True)
        compressor.model.save_pretrained("checkpoints/compressor")
        reconstructor.model.save_pretrained("checkpoints/reconstructor")
        print("Adapters saved to checkpoints/")
    finally:
        profiler.finish()
        print(f"Profiling artifacts: {profiler.run_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--profile", type=int, default=0,
                        metavar="N", help="Profile N steps with torch.profiler (0 = disabled)")
    args = parser.parse_args()
    train(Config.load(), args.data_path, args.steps, args.device,
          use_wandb=args.wandb, profile_steps=args.profile)
