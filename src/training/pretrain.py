"""
Phase 1: Distillation pretraining.

Trains compressor+reconstructor so that CEs are valid soft-token substitutes
for the frozen LLM backbone.

Usage:
  python -m sf.training.pretrain --data-path data/train.txt --steps 500 --device cuda
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


def train(cfg: Config, data_path: Path, steps: int, device: str = "cpu") -> None:
    dev = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(dev)
    for p in llm.parameters():
        p.requires_grad_(False)

    compressor = Compressor(cfg, device=dev)
    reconstructor = Reconstructor(cfg, device=dev)
    fingerprinter = Fingerprinter(str(dev))
    reconstructor.set_fingerprinter(fingerprinter)
    extractor = EntityExtractor(max_entities=cfg.E)
    segmenter = Segmenter(cfg)

    params = list(compressor.parameters()) + list(reconstructor.parameters())
    optimizer = AdamW(params, lr=cfg.pretrain_learning_rate)

    num_layers = llm.config.num_hidden_layers
    layer_range = range(num_layers // 2, num_layers)
    embed = llm.get_input_embeddings()

    text = data_path.read_text(encoding="utf-8") if data_path.exists() else (
        "The quick brown fox jumps over the lazy dog. " * 200  # fallback for testing
    )
    all_ids = tokenizer.encode(text, add_special_tokens=False)
    segments = segmenter.segment(all_ids)
    if not segments:
        print("No segments found."); return

    step = 0
    while step < steps:
        for seg_ids in segments:
            if step >= steps:
                break
            seg_tensor = torch.tensor([seg_ids], device=dev)
            with torch.no_grad():
                orig_embeds = embed(seg_tensor)[0]                 # [N, D]
                out_full = llm(seg_tensor, output_hidden_states=True)

            seg_text = tokenizer.decode(seg_ids)
            fp = fingerprinter.encode(seg_text)
            entities = extractor.extract(seg_text)
            sents = [s.strip() for s in seg_text.split(".") if s.strip()]
            anchors = SanityAnchors(
                boundary_sentences=[sents[0] if sents else "", sents[-1] if sents else ""],
                entities=entities, semantic_fingerprint=fp,
            )

            ce_l2 = compressor.compress(orig_embeds, cfg.C_L2)   # [C_L2, D]
            ce_seq = ce_l2.unsqueeze(0)                           # [1, C_L2, D]

            # L_distill: match hidden states at mid-to-late layers
            out_ce = llm(inputs_embeds=ce_seq, output_hidden_states=True)
            l_distill = torch.tensor(0.0, device=dev)
            for layer_idx in layer_range:
                h_full = out_full.hidden_states[layer_idx][:, -1, :]
                h_ce = out_ce.hidden_states[layer_idx][:, -1, :]
                std = h_full.std().clamp(min=1e-6)
                l_distill = l_distill + F.smooth_l1_loss(h_ce / std, h_full.detach() / std)
            l_distill = l_distill / max(len(layer_range), 1)

            # L_recon: reconstruct -> match original embedding mean
            result = reconstructor.reconstruct(ce_l2, anchors, [], None, "l3_to_l2")
            recon_mean = result.ce_tensor.mean(dim=0)
            orig_mean = orig_embeds.mean(dim=0)
            l_recon = 1.0 - F.cosine_similarity(recon_mean.unsqueeze(0), orig_mean.unsqueeze(0)).mean()

            loss = l_distill + l_recon
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 50 == 0:
                print(f"Step {step}/{steps}  L_distill={l_distill.item():.4f}  L_recon={l_recon.item():.4f}")

    Path("checkpoints").mkdir(exist_ok=True)
    compressor.model.save_pretrained("checkpoints/compressor")
    reconstructor.model.save_pretrained("checkpoints/reconstructor")
    print("Adapters saved to checkpoints/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train(Config.load(), args.data_path, args.steps, args.device)
