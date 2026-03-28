#!/usr/bin/env python3
"""
Weight Outlier Analyzer — Single-command entry point.

Usage:
    python run.py deepseek-ai/DeepSeek-V3
    python run.py openai-community/gpt2 --output-dir ./results
    python run.py meta-llama/Llama-3-8B --token hf_xxx

Takes a HuggingFace model name → runs full pipeline:
    1. analyze.py  — collect per-tensor stats (resumable JSONL)
    2. visualize.py — generate all charts
    3. report.py   — produce markdown report
"""

import argparse
import os
import sys
import time

# Import sibling modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from analyze import run as run_analyze
from visualize import run as run_visualize
from report import run as run_report


def model_slug(model_id: str) -> str:
    """Convert model ID to a filesystem-safe name."""
    return model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Full weight outlier analysis for any HuggingFace model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py openai-community/gpt2
  python run.py deepseek-ai/DeepSeek-V3 --output-dir ./ds_v3_analysis
  python run.py meta-llama/Llama-3-8B --token hf_xxx
        """)

    parser.add_argument("model", help="HuggingFace model ID (e.g. openai-community/gpt2)")
    parser.add_argument("--output-dir", default=None,
                        help="Base output directory (default: ./results/<model_slug>)")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token for gated models")
    parser.add_argument("--skip-analyze", action="store_true",
                        help="Skip analysis step (reuse existing JSONL)")
    parser.add_argument("--skip-visualize", action="store_true",
                        help="Skip visualization step")
    parser.add_argument("--skip-report", action="store_true",
                        help="Skip report generation step")

    args = parser.parse_args()

    slug = model_slug(args.model)
    base_dir = args.output_dir or os.path.join("results", slug)
    os.makedirs(base_dir, exist_ok=True)

    jsonl_path = os.path.join(base_dir, f"{slug}_stats.jsonl")
    images_dir = os.path.join(base_dir, "images")
    report_path = os.path.join(base_dir, f"{slug}_weight_outlier_analysis.md")

    # Friendly model name for chart titles / report heading
    model_name = args.model.split("/")[-1]

    print("=" * 60)
    print(f"  Weight Outlier Analyzer")
    print(f"  Model:  {args.model}")
    print(f"  Output: {base_dir}")
    print("=" * 60)

    t0 = time.time()

    # ── Step 1: Analyze ──
    if not args.skip_analyze:
        print(f"\n{'─'*50}")
        print("STEP 1/3 — Analyzing tensors")
        print(f"{'─'*50}")
        run_analyze(args.model, jsonl_path, token=args.token)
    else:
        print("\n[Skipping analysis — using existing JSONL]")

    if not os.path.exists(jsonl_path):
        print(f"ERROR: No JSONL found at {jsonl_path}")
        sys.exit(1)

    # ── Step 2: Visualize ──
    if not args.skip_visualize:
        print(f"\n{'─'*50}")
        print("STEP 2/3 — Generating visualizations")
        print(f"{'─'*50}")
        run_visualize(jsonl_path, images_dir, model_name=model_name)
    else:
        print("\n[Skipping visualization]")

    # ── Step 3: Report ──
    if not args.skip_report:
        print(f"\n{'─'*50}")
        print("STEP 3/3 — Writing report")
        print(f"{'─'*50}")
        run_report(jsonl_path, images_dir, report_path,
                   model_name=model_name, images_subdir="images")
    else:
        print("\n[Skipping report]")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.0f}s")
    print(f"  Stats:   {jsonl_path}")
    print(f"  Charts:  {images_dir}/")
    print(f"  Report:  {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
