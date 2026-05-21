"""
Usage:
  python eval.py --benchmark niah
  python eval.py --benchmark narrativeqa --max-samples 200
  python eval.py --benchmark longbench --tasks narrativeqa,hotpotqa,gov_report
  python eval.py --benchmark all --max-samples 50 --dry-run
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.config import Config
from eval.runner import ModelRunner
from eval.metrics import token_f1, exact_match, rouge_l
from eval.visualize import niah_heatmap, score_distribution, longbench_bar, overview_summary
from eval.benchmarks import niah as niah_bench
from eval.benchmarks import narrativeqa as nqa_bench
from eval.benchmarks import longbench as lb_bench

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_done_keys(jsonl_path: Path, key_fields: list[str]) -> set[tuple]:
    if not jsonl_path.exists():
        return set()
    done = set()
    for line in jsonl_path.read_text().splitlines():
        try:
            r = json.loads(line)
            done.add(tuple(r[f] for f in key_fields))
        except Exception:
            pass
    return done


def _append(jsonl_path: Path, record: dict) -> None:
    with jsonl_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _load_all(jsonl_path: Path) -> list[dict]:
    return [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]


# ── NIAH ──────────────────────────────────────────────────────────────────────

def run_niah(runner: ModelRunner, out_root: Path, max_samples: int | None) -> dict:
    from tqdm import tqdm

    out_dir = out_root / "niah"
    out_dir.mkdir(exist_ok=True)
    samples_path = out_dir / "samples.jsonl"

    samples = niah_bench.generate_samples()
    if max_samples:
        samples = samples[:max_samples]

    done = _load_done_keys(samples_path, ["context_length_target", "depth", "trial"])
    remaining = [s for s in samples
                 if (s.context_length_target, s.depth, s.trial) not in done]
    logger.info("NIAH: %d total, %d already done, %d to run", len(samples), len(done), len(remaining))

    for sample in tqdm(remaining, desc="NIAH", unit="sample"):
        try:
            prompt = niah_bench.format_prompt(sample)
            prediction, n_input = runner.generate(prompt, max_new_tokens=50)
            record = niah_bench.score_sample(sample, prediction)
            record["n_input_tokens"] = n_input
            _append(samples_path, record)
        except Exception as exc:
            logger.warning("NIAH sample failed: %s", exc)

    all_results = _load_all(samples_path)
    if not all_results:
        return {}

    # Build grid: depth → context_length → list[hit]
    grid_raw: dict[float, dict[int, list[float]]] = {}
    for r in all_results:
        d, l, h = r["depth"], r["context_length_target"], r["hit"]
        grid_raw.setdefault(d, {}).setdefault(l, []).append(h)

    grid_means = {d: {l: sum(v) / len(v) for l, v in lmap.items()}
                  for d, lmap in grid_raw.items()}
    overall = sum(r["hit"] for r in all_results) / len(all_results)

    summary = {"overall_hit_rate": overall, "grid": grid_means, "n_samples": len(all_results)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    niah_heatmap(grid_means, out_dir / "heatmap.png")
    logger.info("NIAH done — overall hit rate: %.1f%%  →  %s", overall * 100, out_dir)
    return summary


# ── NarrativeQA ───────────────────────────────────────────────────────────────

def run_narrativeqa(runner: ModelRunner, out_root: Path, max_samples: int | None) -> dict:
    from tqdm import tqdm

    out_dir = out_root / "narrativeqa"
    out_dir.mkdir(exist_ok=True)
    samples_path = out_dir / "samples.jsonl"

    logger.info("Loading NarrativeQA (deepmind/narrativeqa) …")
    samples = nqa_bench.load_samples(max_samples=max_samples)

    done = _load_done_keys(samples_path, ["doc_id"])
    remaining = [s for s in samples if s.doc_id not in done]
    logger.info("NarrativeQA: %d total, %d already done, %d to run",
                len(samples), len(done), len(remaining))

    for sample in tqdm(remaining, desc="NarrativeQA", unit="sample"):
        try:
            prediction, n_input = runner.generate(sample.format_prompt(), max_new_tokens=100)
            f1 = token_f1(prediction, sample.references)
            _append(samples_path, {
                "doc_id": sample.doc_id,
                "question": sample.question,
                "references": sample.references,
                "prediction": prediction,
                "f1": f1,
                "n_input_tokens": n_input,
            })
        except Exception as exc:
            logger.warning("NarrativeQA sample %s failed: %s", sample.doc_id, exc)

    all_results = _load_all(samples_path)
    if not all_results:
        return {}

    f1_scores = [r["f1"] for r in all_results]
    summary = {
        "mean_f1": sum(f1_scores) / len(f1_scores),
        "median_f1": float(sorted(f1_scores)[len(f1_scores) // 2]),
        "n_samples": len(f1_scores),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    score_distribution(f1_scores, "Token F1", out_dir / "distribution.png",
                       title="NarrativeQA Score Distribution")
    logger.info("NarrativeQA done — mean F1: %.3f  →  %s", summary["mean_f1"], out_dir)
    return summary


# ── LongBench ─────────────────────────────────────────────────────────────────

def _score_lb(prediction: str, references: list[str], metric: str) -> float:
    if metric == "f1":
        return token_f1(prediction, references)
    if metric == "rouge_l":
        return rouge_l(prediction, references)
    if metric == "exact_match":
        return exact_match(prediction, references)
    return token_f1(prediction, references)


def run_longbench(
    runner: ModelRunner,
    out_root: Path,
    tasks: list[str],
    max_samples: int | None,
) -> dict:
    from tqdm import tqdm

    lb_dir = out_root / "longbench"
    lb_dir.mkdir(exist_ok=True)

    task_summaries: dict[str, float] = {}

    for task in tasks:
        metric = lb_bench.TASK_METRIC.get(task, "f1")
        task_dir = lb_dir / task
        task_dir.mkdir(exist_ok=True)
        samples_path = task_dir / "samples.jsonl"

        logger.info("Loading LongBench task: %s …", task)
        try:
            samples = lb_bench.load_samples(task, max_samples=max_samples)
        except Exception as exc:
            logger.warning("Failed to load LongBench/%s: %s", task, exc)
            continue

        done = _load_done_keys(samples_path, ["sample_id"])
        remaining = [s for s in samples if s.sample_id not in done]
        logger.info("LB/%s: %d total, %d already done, %d to run",
                    task, len(samples), len(done), len(remaining))

        for sample in tqdm(remaining, desc=f"LB/{task}", unit="sample"):
            try:
                prediction, n_input = runner.generate(sample.format_prompt(), max_new_tokens=200)
                score = _score_lb(prediction, sample.references, metric)
                _append(samples_path, {
                    "sample_id": sample.sample_id,
                    "task": task,
                    "query": sample.query,
                    "references": sample.references,
                    "prediction": prediction,
                    "score": score,
                    "metric": metric,
                    "n_input_tokens": n_input,
                })
            except Exception as exc:
                logger.warning("LB/%s sample %s failed: %s", task, sample.sample_id, exc)

        all_results = _load_all(samples_path)
        if not all_results:
            continue

        scores = [r["score"] for r in all_results]
        mean_score = sum(scores) / len(scores)
        task_summary = {"mean_score": mean_score, "metric": metric, "n_samples": len(scores)}
        (task_dir / "summary.json").write_text(json.dumps(task_summary, indent=2))

        task_summaries[task] = mean_score
        logger.info("LB/%s done — mean %s: %.3f", task, metric, mean_score)

    if task_summaries:
        (lb_dir / "overview.json").write_text(json.dumps(task_summaries, indent=2))
        longbench_bar(task_summaries, lb_dir / "scores.png")

    return task_summaries


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline (full context) on long-context benchmarks.")
    parser.add_argument("--benchmark", nargs="+",
                        choices=["niah", "narrativeqa", "longbench", "all"], default=["all"],
                        help="Which benchmark(s) to run.")
    parser.add_argument("--tasks", default=",".join(lb_bench.DEFAULT_TASKS),
                        help="Comma-separated LongBench task names (default: %(default)s).")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model ID (default: from config.yaml).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-input-tokens", type=int, default=32768,
                        help="Hard cap on input token count; tail is kept (preserves the question).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per benchmark (useful for quick sanity-checks).")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-name", default=None,
                        help="Override the auto-generated run directory name.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip model loading; verify data pipelines only.")
    args = parser.parse_args()

    benchmarks = set(args.benchmark)
    if "all" in benchmarks:
        benchmarks = {"niah", "narrativeqa", "longbench"}

    cfg = Config.load()
    model_name = args.model or cfg.model_name

    run_name = args.run_name or f"baseline_full_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root = Path(args.output_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_name": run_name,
        "model": model_name,
        "device": args.device,
        "benchmarks": sorted(benchmarks),
        "longbench_tasks": args.tasks.split(","),
        "max_samples": args.max_samples,
        "max_input_tokens": args.max_input_tokens,
        "dry_run": args.dry_run,
        "timestamp": datetime.now().isoformat(),
    }
    (out_root / "config.json").write_text(json.dumps(run_config, indent=2))
    logger.info("Run: %s", run_name)
    logger.info("Output: %s", out_root)

    runner = ModelRunner(
        model_name=model_name,
        device=args.device,
        max_input_tokens=args.max_input_tokens,
        dry_run=args.dry_run,
    )

    all_summaries: dict = {}

    if "niah" in benchmarks:
        all_summaries["niah"] = run_niah(runner, out_root, args.max_samples)

    if "narrativeqa" in benchmarks:
        all_summaries["narrativeqa"] = run_narrativeqa(runner, out_root, args.max_samples)

    if "longbench" in benchmarks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        all_summaries["longbench"] = run_longbench(runner, out_root, tasks, args.max_samples)

    overview_summary(all_summaries, out_root / "overview.png")
    (out_root / "all_summaries.json").write_text(json.dumps(all_summaries, indent=2))

    print(f"\n{'─' * 50}")
    print(f"Results saved to: {out_root}")
    if "niah" in all_summaries and all_summaries["niah"]:
        print(f"  NIAH hit rate     : {all_summaries['niah'].get('overall_hit_rate', 0):.1%}")
    if "narrativeqa" in all_summaries and all_summaries["narrativeqa"]:
        print(f"  NarrativeQA F1    : {all_summaries['narrativeqa'].get('mean_f1', 0):.3f}")
    if "longbench" in all_summaries:
        for task, score in all_summaries["longbench"].items():
            print(f"  LB/{task:<20}: {score:.3f}")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    main()
