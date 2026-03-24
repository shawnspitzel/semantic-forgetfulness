"""
Download and mix pretraining data for Semantic Forgetfulness.

Usage:
    python scripts/preprocess_data.py              # full corpus (~50M tokens)
    python scripts/preprocess_data.py --small      # ~1M tokens, faster
    python scripts/preprocess_data.py --test       # skip download, use fallback

Output: data/train.txt
"""

import argparse
import os

OUTPUT_PATH = "data/train.txt"
SMALL_DOC_LIMIT = 5_000    # ~1M tokens
FULL_DOC_LIMIT  = 50_000   # ~50M tokens


def get_text(ex: dict) -> str:
    return ex.get("text") or ex.get("content") or ex.get("output") or ""


def download_small():
    """Wikipedia only — quick smoke-test corpus."""
    from datasets import load_dataset

    print("Downloading Wikipedia (small run, ~5000 docs)...")
    ds = load_dataset("wikimedia/wikipedia", "20220301.en", split="train", streaming=True)

    os.makedirs("data", exist_ok=True)
    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in ds:
            if len(ex["text"]) > 500:
                f.write(ex["text"] + "\n\n")
                count += 1
            if count >= SMALL_DOC_LIMIT:
                break

    print(f"Wrote {count} documents to {OUTPUT_PATH}")


def download_full():
    """
    Full recommended mix from pretraining-guide.md §2.5:
      40% OpenWebText2, 25% Wikipedia, 20% UltraChat, 10% PG-19, 5% NarrativeQA
    """
    from datasets import load_dataset, interleave_datasets

    print("Loading datasets (streaming)...")
    wiki = load_dataset("wikimedia/wikipedia", "20220301.en",
                        split="train", streaming=True)
    owt  = load_dataset("EleutherAI/pile", data_files="*openwebtext2*",
                        split="train", streaming=True)
    chat = load_dataset("HuggingFaceH4/ultrachat_200k",
                        split="train_sft", streaming=True)
    pg19 = load_dataset("emozilla/pg19",
                        split="train", streaming=True)
    nqa  = load_dataset("deepmind/narrativeqa",
                        split="train", streaming=True)

    dataset = interleave_datasets(
        [owt, wiki, chat, pg19, nqa],
        probabilities=[0.40, 0.25, 0.20, 0.10, 0.05],
        stopping_strategy="first_exhausted",
    )

    os.makedirs("data", exist_ok=True)
    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in dataset:
            t = get_text(ex)
            if len(t) > 500:
                f.write(t + "\n\n")
                count += 1
                if count % 1000 == 0:
                    print(f"  {count} / {FULL_DOC_LIMIT} documents...")
            if count >= FULL_DOC_LIMIT:
                break

    print(f"Wrote {count} documents to {OUTPUT_PATH}")


def write_fallback():
    """No download — write the fallback string so the pipeline at least runs."""
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("The quick brown fox jumps over the lazy dog. " * 2000)
    print(f"Wrote fallback string to {OUTPUT_PATH} (pipeline test only, not real training data)")


def check_output():
    if not os.path.exists(OUTPUT_PATH):
        print("ERROR: output file not found")
        return
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        head = f.read(200)
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Size  : {size_mb:.1f} MB")
    print(f"Head  : {head[:120]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining data")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--small", action="store_true",
                       help="Small run: Wikipedia only, ~5K docs (~1M tokens)")
    group.add_argument("--test", action="store_true",
                       help="No download: write fallback string for pipeline testing only")
    args = parser.parse_args()

    if args.test:
        write_fallback()
    elif args.small:
        download_small()
    else:
        download_full()

    check_output()
