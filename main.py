#!/usr/bin/env python3
"""
misfits — Weight outlier analysis for LLMs.

Master entry point. Runs analysis for all models and generates
the cross-model comparison report.

Usage:
    python main.py                    # run all models
    python main.py gpt2               # run specific model
    python main.py --skip-stats       # reuse existing stats, regenerate charts + reports
    python main.py --comparison-only  # only regenerate the comparison report
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# Add repo root to path so shared/ imports work from model analyze.py files
sys.path.insert(0, str(ROOT))

# ─── Model registry ───────────────────────────────────────────────────────────

MODELS = {
    "gpt2": {
        "name": "GPT-2",
        "dir": ROOT / "models" / "gpt2",
        "report": "gpt2_weight_outlier_analysis.md",
    },
    "llama-3.2-1b": {
        "name": "Llama 3.2 1B",
        "dir": ROOT / "models" / "llama-3.2-1b",
        "report": "llama_3_2_1b_weight_outlier_analysis.md",
    },
    "qwen3.5-0.8b": {
        "name": "Qwen3.5-0.8B",
        "dir": ROOT / "models" / "qwen3.5-0.8b",
        "report": "qwen3_5_0_8b_weight_outlier_analysis.md",
    },
}

# ─── Model import ─────────────────────────────────────────────────────────────

def import_model(key: str):
    """Import a model's analyze module by key.

    Uses spec_from_file_location instead of regular imports because
    model directory names contain hyphens and dots.
    """
    analyze_path = MODELS[key]["dir"] / "analyze.py"
    if not analyze_path.exists():
        raise FileNotFoundError(f"No analyze.py found for model '{key}' at {analyze_path}")
    spec = importlib.util.spec_from_file_location(f"models_{key}_analyze", analyze_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Comparison report ────────────────────────────────────────────────────────

def _load_summary(model_key: str) -> dict | None:
    """Load the summary JSON for a model, returning None if not found."""
    info = MODELS[model_key]
    # Convention: *_summary.json or *_weight_outlier_analysis_summary.json
    candidates = list(info["dir"].glob("*_summary.json"))
    if not candidates:
        return None
    # Prefer the most recently modified if multiple exist
    summary_path = max(candidates, key=lambda p: p.stat().st_mtime)
    with open(summary_path) as f:
        d = json.load(f)
    d["_name"] = info["name"]
    d["_key"] = model_key
    return d


def _fmt(val, fmt: str) -> str:
    """Format a value with the given format spec, handling % suffix."""
    if val is None:
        return "—"
    if fmt.endswith("%"):
        return f"{val:{fmt[:-1]}}%"
    return f"{val:{fmt}}"


def generate_comparison(model_keys: list[str]) -> None:
    """Generate the cross-model comparison report.

    Reads each model's summary JSON to build models/comparison.md.
    Includes architecture metadata, key metrics, per-component breakdowns,
    systematic outlier dimensions, generational analysis, and links.

    Args:
        model_keys: List of model keys (subset of MODELS) to include.
    """
    from shared.report import format_number

    # Load summaries — skip models with no summary yet
    summaries: dict[str, dict] = {}
    for key in model_keys:
        s = _load_summary(key)
        if s is not None:
            summaries[key] = s
        else:
            print(f"  Warning: no summary JSON found for '{key}', skipping from comparison")

    if len(summaries) < 2:
        print("  Need at least 2 model summaries for comparison — skipping comparison report")
        return

    # Architecture metadata (static, curated per model)
    ARCH_META = {
        "gpt2": {
            "architecture": "Dense Transformer",
            "params": "124M",
            "layers": "12",
            "hidden_dim": "768",
            "release": "2019",
        },
        "llama-3.2-1b": {
            "architecture": "Dense Transformer (GQA)",
            "params": "1.2B",
            "layers": "16",
            "hidden_dim": "2048",
            "release": "2024",
        },
        "qwen3.5-0.8b": {
            "architecture": "Hybrid: Gated DeltaNet + Attention",
            "params": "818M (619M analyzed)",
            "layers": "24",
            "hidden_dim": "1024",
            "release": "2026",
        },
    }

    lines: list[str] = []
    w = lines.append

    # ── Header ────────────────────────────────────────────────────────────────
    w("# Weight Outlier Comparison Across Models\n")
    w("A cross-model comparison of weight distribution outlier patterns.\n")

    # ── Models table ──────────────────────────────────────────────────────────
    w("## Models Analyzed\n")

    col_names = [summaries[k]["_name"] for k in summaries]
    header = "| |" + "|".join(f" {n} " for n in col_names) + "|"
    sep    = "|---|" + "|".join("---|" for _ in summaries)[:-1] + "|"
    w(header)
    w(sep)

    def meta_row(label: str, meta_key: str) -> str:
        row = f"| **{label}** |"
        for k in summaries:
            val = ARCH_META.get(k, {}).get(meta_key, "—")
            row += f" {val} |"
        return row

    def stat_row(label: str, stat_key: str) -> str:
        row = f"| **{label}** |"
        for k, s in summaries.items():
            val = s.get(stat_key)
            row += f" {val if val is not None else '—'} |"
        return row

    w(meta_row("Parameters", "params"))
    w(meta_row("Architecture", "architecture"))
    w(meta_row("Layers", "layers"))
    w(meta_row("Hidden dim", "hidden_dim"))

    # Tensors analyzed from summary
    row = "| **Tensors analyzed** |"
    for s in summaries.values():
        row += f" {s.get('total_tensors', '?')} |"
    w(row)

    w(meta_row("Release", "release"))
    w("")

    # ── Key metrics table ──────────────────────────────────────────────────────
    w("## Key Metrics\n")

    metrics = [
        ("Global |max weight|",       "global_abs_max",              ".2f"),
        ("Median |max weight|",        "median_abs_max",              ".2f"),
        ("Peak kurtosis",              "kurtosis_max",                ".1f"),
        ("Median kurtosis",            "kurtosis_median",             ".2f"),
        ("Mean kurtosis",              "kurtosis_mean",               ".1f"),
        ("P99 kurtosis",               "kurtosis_p99",                ".1f"),
        ("Median dynamic range",       "dynamic_range_median",        ".1f"),
        ("Max dynamic range",          "dynamic_range_max",           ".1f"),
        ("Mean % beyond 3σ",           "mean_beyond_3sigma_pct",      ".3f%"),
        ("Mean % beyond 10σ",          "mean_beyond_10sigma_pct",     ".4f%"),
        ("Mean INT4 clip improvement", "mean_clip_improvement_pct",   ".1f%"),
        ("Max INT4 clip improvement",  "max_clip_improvement_pct",    ".1f%"),
    ]

    w("| Metric |" + "|".join(f" {s['_name']} " for s in summaries.values()) + "|")
    w("|--------|" + "|".join("---:|" for _ in summaries)[:-1] + "|")

    for label, key, fmt in metrics:
        row = f"| **{label}** |"
        for s in summaries.values():
            val = s.get(key)
            row += f" {_fmt(val, fmt)} |"
        w(row)
    w("")

    # ── Component comparison ───────────────────────────────────────────────────
    w("## Component Comparison\n")

    # Model-specific narrative summaries (curated)
    COMPONENT_NARRATIVE = {
        "gpt2": (
            "GPT-2's outlier problem is dominated by **MLP layers**, especially `c_proj` in "
            "early layers (layers 0–3). These have kurtosis 400–800, driven by a few extreme "
            "output channels."
        ),
        "qwen3.5-0.8b": (
            "Standard attention and MLP are extremely well-behaved. The **Gated DeltaNet layers** "
            "have moderate kurtosis. A single `mtp.fc` tensor has kurtosis 712 — an isolated "
            "extreme outlier."
        ),
        "llama-3.2-1b": (
            "Llama 3.2 is **remarkably clean** — no tensor exceeds kurtosis 16.1, and the global "
            "max weight is only 1.23. This is the most quantization-friendly model of the three "
            "by a wide margin."
        ),
    }

    for key, s in summaries.items():
        w(f"### {s['_name']}\n")
        comp_stats = s.get("component_stats", {})
        if comp_stats:
            w("| Component | Count | Mean Kurtosis | Max Kurtosis | Max \\|weight\\| |")
            w("|-----------|------:|--------------:|-------------:|--------------:|")
            for comp, cs in sorted(comp_stats.items()):
                # Bold the maximum kurtosis and max weight cells
                max_k = cs["max_kurtosis"]
                max_a = cs["max_abs_max"]
                # Find global max for this model
                all_max_k = max(c["max_kurtosis"] for c in comp_stats.values())
                all_max_a = max(c["max_abs_max"]   for c in comp_stats.values())
                k_str = f"**{max_k:.1f}**" if max_k == all_max_k else f"{max_k:.1f}"
                a_str = f"**{max_a:.2f}**" if max_a == all_max_a else f"{max_a:.4f}"
                w(f"| {comp} | {cs['count']} | {cs['mean_kurtosis']:.1f} | {k_str} | {a_str} |")
        narrative = COMPONENT_NARRATIVE.get(key)
        if narrative:
            w(f"\n{narrative}\n")
        else:
            w("")

    # ── Systematic outlier dimensions ──────────────────────────────────────────
    w("## Systematic Outlier Dimensions\n")

    dim_header = "| |" + "|".join(f" {s['_name']} " for s in summaries.values()) + "|"
    dim_sep    = "|---|" + "|".join("---|" for _ in summaries)[:-1] + "|"
    w(dim_header)
    w(dim_sep)

    def top_dim_str(s: dict, idx: int) -> str:
        dims = s.get("top_outlier_dims", [])
        if idx < len(dims):
            d = dims[idx]
            return f"dim {d['dim']} ({d['pct']:.0f}%)"
        return "—"

    def outlier_pattern(s: dict) -> str:
        dims = s.get("top_outlier_dims", [])
        if not dims:
            return "—"
        top_pct = dims[0]["pct"] if dims else 0
        if top_pct > 40:
            return "Sharply concentrated"
        elif top_pct > 20:
            return "Moderately diffuse"
        return "Diffuse"

    top_row = "| **Top outlier dim** |"
    for s in summaries.values():
        top_row += f" {top_dim_str(s, 0)} |"
    w(top_row)

    runner_row = "| **Runner-ups** |"
    for s in summaries.values():
        runners = ", ".join(top_dim_str(s, i) for i in range(1, 3))
        runner_row += f" {runners} |"
    w(runner_row)

    pattern_row = "| **Pattern** |"
    for s in summaries.values():
        pattern_row += f" {outlier_pattern(s)} |"
    w(pattern_row)
    w("")
    w("Newer models show progressively less concentrated outlier dimensions — likely a sign of "
      "improved training practices.\n")

    # ── Generational analysis ──────────────────────────────────────────────────
    w("## Evolution of Quantization Friendliness\n")
    w("Clear generational improvement:\n")
    w("1. **GPT-2 (2019)**: Severe outliers. MLP c_proj layers are quantization disasters "
      "(kurtosis ~800). Requires outlier-aware quantization (GPTQ, SmoothQuant, etc.)\n")
    w("2. **Qwen3.5-0.8B (2026)**: Mixed. Standard transformer components are near-Gaussian. "
      "But novel architecture components (DeltaNet, MTP) introduce new outlier patterns that "
      "need fresh quantization strategies.\n")
    w("3. **Llama 3.2 1B (2024)**: Excellent. Near-Gaussian everywhere, max kurtosis only 16. "
      "Almost any quantization method should work well. Likely benefits from Meta's "
      "quantization-aware training and architectural improvements.\n")

    # ── Detailed report links ─────────────────────────────────────────────────
    w("## Detailed Reports\n")
    for key, info in MODELS.items():
        if key not in summaries:
            continue
        preferred_report = info["dir"] / info["report"]
        if preferred_report.exists():
            rel = preferred_report.relative_to(ROOT / "models")
            w(f"- [{info['name']}]({rel})")
            continue

        fallback_reports = sorted(info["dir"].glob("*_weight_outlier_analysis.md"))
        if fallback_reports:
            rel = fallback_reports[0].relative_to(ROOT / "models")
            w(f"- [{info['name']}]({rel})")
    w("")

    comparison_path = ROOT / "models" / "comparison.md"
    comparison_path.write_text("\n".join(lines))
    print(f"  Comparison report: {comparison_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="misfits — Weight outlier analysis for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=list(MODELS.keys()),
        help=f"Models to analyze (default: all). Choices: {', '.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip stats collection, reuse existing JSONL; regenerate charts + reports",
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Only regenerate the cross-model comparison report (no per-model analysis)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace access token for gated models",
    )

    args = parser.parse_args()

    # Validate model keys up-front
    unknown = [k for k in args.models if k not in MODELS]
    if unknown:
        parser.error(
            f"Unknown model(s): {', '.join(unknown)}. "
            f"Available: {', '.join(MODELS.keys())}"
        )

    t0 = time.time()
    print("=" * 60)
    print("  misfits — Weight Outlier Analysis for LLMs")
    print("=" * 60)

    if not args.comparison_only:
        for key in args.models:
            print(f"\n{'─' * 60}")
            print(f"  Analyzing: {MODELS[key]['name']}")
            print(f"{'─' * 60}")

            mod = import_model(key)
            mod.main(skip_stats=args.skip_stats, token=args.token)

    print(f"\n{'─' * 60}")
    print("  Generating cross-model comparison")
    print(f"{'─' * 60}")
    generate_comparison(args.models)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  All done in {elapsed:.0f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
