"""Simulate the chatbot input loop with direct stdin lines."""
import sys
import io

# Simulate piped input
test_input = "Hello world.\n/stats\n/quit\n"
sys.stdin = io.StringIO(test_input)

from sf.config import Config
from sf.inference_loop import InferenceLoop

def print_stats(loop):
    s = loop.hit_rate_stats
    total = s["total_segments"]
    l1_rate = s["l1_count"] / total if total else 0.0
    print(f"\n── Cache Stats ──────────────────────────────")
    print(f"  Segments  : {total}  (L1={s['l1_count']} L2={s['l2_count']} L3={s['l3_count']})")
    print(f"  L1 rate   : {l1_rate:.1%}")
    print(f"  Misses    : {s['miss_events']}")
    print(f"─────────────────────────────────────────────\n")

cfg = Config.load()
loop = InferenceLoop(cfg)

print("Semantic Forgetfulness (test mode)")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("[EOF reached]")
        break

    print(f"[Got input: {repr(user_input)}]")
    if not user_input:
        continue
    if user_input == "/quit":
        print("[/quit received]")
        break
    if user_input == "/stats":
        print_stats(loop)
        continue
    if user_input == "/done":
        print_stats(loop)
        break

    loop.process_text(user_input)
    response = loop.generate_response(user_input)
    print(f"Assistant: {response}")

print("Exit clean.")
