from sf.config import Config
from sf.inference_loop import InferenceLoop

cfg = Config.load()
loop = InferenceLoop(cfg)
for text in [
    'Attention mechanisms allow models to focus on relevant tokens.',
    'Long context windows cause context rot as attention dilutes.',
    'The L1 cache holds the active token window on GPU VRAM.',
    'Cache misses are training signals for the LoRA compressor.',
]:
    loop.process_text(text)

stats = loop.hit_rate_stats
print(stats)
assert stats['total_segments'] >= 4, f'Expected >=4 segments, got {stats["total_segments"]}'
print('Integration check: PASSED')
