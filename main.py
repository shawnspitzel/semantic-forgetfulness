#!/usr/bin/env python3
"""
Semantic Forgetfulness — terminal chatbot MVP.

Default model: meta-llama/Llama-3.2-3B-Instruct
Requires: huggingface-cli login (gated model)

Usage:
  python main.py                         # mock mode (no model download)
  python main.py --load-models           # real Llama inference
  python main.py --model gpt2 --load-models  # override model

Commands:
  /stats    print cache hit-rate stats
  /done     end session + fine-tune
  /quit     exit without fine-tuning
"""
from __future__ import annotations
import argparse, sys, signal
from pathlib import Path

import torch

from sf.config import Config
from sf.inference_loop import InferenceLoop
from sf.training.finetune import run_finetune
from sf.data_structures import FullMissEvent


def print_stats(loop: InferenceLoop) -> None:
    s = loop.hit_rate_stats
    total = s["total_segments"]
    l1_rate = s["l1_count"] / total if total else 0.0
    print(f"\n── Cache Stats ──────────────────────────────")
    print(f"  Segments  : {total}  (L1={s['l1_count']} L2={s['l2_count']} L3={s['l3_count']})")
    print(f"  L1 rate   : {l1_rate:.1%}")
    print(f"  Misses    : {s['miss_events']}")
    print(f"─────────────────────────────────────────────\n")


def end_session(loop: InferenceLoop, cfg: Config) -> None:
    print("\n[Session ending...]")
    if loop._compressor is None:
        print(f"[Mock mode — {len(loop.cache_controller.miss_log)} misses logged, no fine-tuning.]")
        return

    # Reconcile miss log -> FullMissEvents
    full_events: list[FullMissEvent] = []
    W = cfg.context_window_W
    all_embeddings = loop._embedding_history  # list of [N, D] tensors

    for event in loop.cache_controller.miss_log:
        if event.miss_type == "total":
            continue
        meta = loop.cache_controller.get_metadata(event.segment_id)
        if meta is None:
            continue

        # Approximate context window from embedding history
        ctx_list = [e.to(loop.device) for e in all_embeddings]
        if not ctx_list:
            continue
        ctx = torch.cat(ctx_list, dim=0)
        pos = meta.source_position
        ctx_window = ctx[max(0, pos - W): min(ctx.shape[0], pos + W)]

        if event.miss_type == "soft":
            if not ctx_list:
                continue
            seg_emb = ctx_list[0]
            ce = loop._compressor.compress(seg_emb, cfg.C_L2).detach()
            full_events.append(FullMissEvent(
                segment_input=seg_emb, context_window=ctx_window,
                ce_produced=ce, miss_level="l2", session_position=pos,
            ))
        else:
            l3e = loop.cache_controller._l3.get(event.segment_id)
            if l3e is None:
                continue
            full_events.append(FullMissEvent(
                segment_input=l3e.l2_ce_at_demotion, context_window=ctx_window,
                ce_produced=l3e.concept_embeddings.detach(),
                miss_level="l3", session_position=pos,
            ))

    result = run_finetune(cfg, full_events, loop._compressor, loop._reconstructor, loop.device)

    if result["skipped"]:
        print(f"[Fine-tuning skipped: {result['reason']} ({result['count']} events)]")
    else:
        print(f"[Fine-tuning: {result['miss_events_used']} events, loss={result['avg_loss']:.4f}]")
        Path("checkpoints").mkdir(exist_ok=True)
        loop._compressor.model.save_pretrained("checkpoints/compressor")
        loop._reconstructor.model.save_pretrained("checkpoints/reconstructor")
        print("[Adapters saved to checkpoints/]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Override model (default: meta-llama/Llama-3.2-3B-Instruct)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--load-models", action="store_true",
                        help="Load LLM for real inference (requires HF login for Llama)")
    args = parser.parse_args()

    cfg = Config.load()
    if args.model:
        # Override model_name by creating new Config with updated value
        import dataclasses
        cfg = dataclasses.replace(cfg, model_name=args.model)
    loop = InferenceLoop(cfg, device=args.device, load_models=args.load_models)

    print("Semantic Forgetfulness")
    print(f"  Model : {cfg.model_name}")
    print(f"  Device: {args.device}  |  Full inference: {args.load_models}")
    print("  Commands: /stats  /done  /quit\n")

    signal.signal(signal.SIGTERM, lambda *_: (end_session(loop, cfg), sys.exit(0)))

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            end_session(loop, cfg)
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/stats":
            print_stats(loop); continue
        if user_input == "/done":
            end_session(loop, cfg); print_stats(loop); break

        loop.process_text(user_input)
        print(f"Assistant: {loop.generate_response(user_input)}")


if __name__ == "__main__":
    main()
