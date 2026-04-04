"""Report generation: model-level summary computation and markdown output."""

from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np


# ────────────────────────────────────────
# Summary computation
# ────────────────────────────────────────

def compute_summary(records: list[dict]) -> dict:
    """Compute model-level aggregate statistics from per-tensor records.

    Aggregates kurtosis, absolute maximum, dynamic range, sigma outlier
    percentages, INT4 quantization improvement, per-component breakdowns,
    and the top-10 worst tensors by kurtosis.

    Args:
        records: List of per-tensor stat dicts as produced by
                 :func:`shared.stats.compute_stats` merged with
                 :func:`shared.stats.classify` metadata.

    Returns:
        A summary dict with keys including:
            - ``total_tensors``, ``component_counts``, ``total_params_analyzed``
            - ``kurtosis_mean``, ``kurtosis_median``, ``kurtosis_max``,
              ``kurtosis_p99``
            - ``global_abs_max``, ``median_abs_max``
            - ``dynamic_range_median``, ``dynamic_range_max``
            - ``mean_beyond_Nsigma_pct`` (for N in 3, 5, 8, 10)
            - ``mean_clip_improvement_pct``, ``max_clip_improvement_pct``
            - ``worst_10`` — list of top-10 tensors by kurtosis
            - ``component_stats`` — per-component aggregates
            - ``top_outlier_dims`` — most frequently flagged input dimensions
    """
    summary: dict = {}

    # ── Basic counts ──────────────────────────────────────────────────────────
    summary["total_tensors"] = len(records)
    components: dict[str, int] = defaultdict(int)
    for r in records:
        components[r.get("component", "other")] += 1
    summary["component_counts"] = dict(components)
    summary["total_params_analyzed"] = sum(r.get("num_params", 0) for r in records)

    # ── Kurtosis ──────────────────────────────────────────────────────────────
    kurtosis_vals = [r["excess_kurtosis"] for r in records if "excess_kurtosis" in r]
    if kurtosis_vals:
        summary["kurtosis_mean"] = float(np.mean(kurtosis_vals))
        summary["kurtosis_median"] = float(np.median(kurtosis_vals))
        summary["kurtosis_max"] = float(np.max(kurtosis_vals))
        summary["kurtosis_p99"] = float(np.percentile(kurtosis_vals, 99))

    # ── Absolute max ──────────────────────────────────────────────────────────
    abs_maxes = [r["abs_max"] for r in records if "abs_max" in r]
    if abs_maxes:
        summary["global_abs_max"] = float(max(abs_maxes))
        summary["median_abs_max"] = float(np.median(abs_maxes))

    # ── Dynamic range ─────────────────────────────────────────────────────────
    dr_vals = [
        r["dynamic_range"]
        for r in records
        if "dynamic_range" in r and r["dynamic_range"] < float("inf")
    ]
    if dr_vals:
        summary["dynamic_range_median"] = float(np.median(dr_vals))
        summary["dynamic_range_max"] = float(max(dr_vals))

    # ── Sigma outlier percentages ─────────────────────────────────────────────
    for s in [3, 5, 8, 10]:
        key = f"beyond_{s}sigma_pct"
        vals = [r[key] for r in records if key in r]
        if vals:
            summary[f"mean_beyond_{s}sigma_pct"] = float(np.mean(vals))

    # ── INT4 quantization improvement ─────────────────────────────────────────
    imp_vals = [
        r["int4_clip_improvement_pct"]
        for r in records
        if "int4_clip_improvement_pct" in r
    ]
    if imp_vals:
        summary["mean_clip_improvement_pct"] = float(np.mean(imp_vals))
        summary["max_clip_improvement_pct"] = float(max(imp_vals))

    # ── Worst tensors by kurtosis ─────────────────────────────────────────────
    ranked = sorted(records, key=lambda r: r.get("excess_kurtosis", 0), reverse=True)
    summary["worst_10"] = [
        {
            "name": r["tensor_name"],
            "kurtosis": r.get("excess_kurtosis", 0),
            "abs_max": r.get("abs_max", 0),
            "component": r.get("component", "unknown"),
        }
        for r in ranked[:10]
    ]

    # ── Per-component aggregates ──────────────────────────────────────────────
    comp_agg: dict[str, dict] = {}
    for comp in components:
        comp_recs = [r for r in records if r.get("component") == comp]
        k_vals = [r["excess_kurtosis"] for r in comp_recs if "excess_kurtosis" in r]
        a_vals = [r["abs_max"] for r in comp_recs if "abs_max" in r]
        comp_agg[comp] = {
            "count": len(comp_recs),
            "mean_kurtosis": float(np.mean(k_vals)) if k_vals else 0.0,
            "max_kurtosis": float(max(k_vals)) if k_vals else 0.0,
            "mean_abs_max": float(np.mean(a_vals)) if a_vals else 0.0,
            "max_abs_max": float(max(a_vals)) if a_vals else 0.0,
        }
    summary["component_stats"] = comp_agg

    # ── Systematic outlier input dimensions ───────────────────────────────────
    dim_freq: dict[int, int] = defaultdict(int)
    total_with_dims = 0
    for r in records:
        dims = r.get("input_dim_outlier_indices", [])
        if dims:
            total_with_dims += 1
            for d in dims:
                dim_freq[d] += 1
    if dim_freq:
        top_dims = sorted(dim_freq.items(), key=lambda x: -x[1])[:10]
        summary["top_outlier_dims"] = [
            {"dim": d, "freq": c, "pct": c / total_with_dims * 100}
            for d, c in top_dims
        ]

    return summary


# ────────────────────────────────────────
# Number formatting
# ────────────────────────────────────────

def format_number(n: float) -> str:
    """Format a number with SI suffix for compact display.

    Examples:
        >>> format_number(1_234_567_890)
        '1.2B'
        >>> format_number(45_000)
        '45.0K'
        >>> format_number(3.14159)
        '3.142'

    Args:
        n: Numeric value to format.

    Returns:
        A compact string representation.
    """
    if abs(n) >= 1e9:
        return f"{n / 1e9:.1f}B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.1f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.4g}"


# ────────────────────────────────────────
# Markdown generation
# ────────────────────────────────────────

def generate_markdown(
    summary: dict,
    model_name: str,
    images_subdir: str,
    chart_manifest: dict[str, str],
) -> str:
    """Generate a full markdown report from computed summary statistics.

    The output is Obsidian-compatible: images are embedded with
    ``![[subdir/filename]]`` syntax. The report includes an overview,
    auto-generated key findings, chart embeds with descriptions, component
    breakdown tables, worst-tensor tables, and a methodology section.

    Args:
        summary:        Dict from :func:`compute_summary`.
        model_name:     Human-readable model name for headings and titles.
        images_subdir:  Subdirectory name used in image embed paths
                        (e.g. ``"images"``).
        chart_manifest: Ordered dict mapping chart display names to filenames,
                        as saved by the visualization step.

    Returns:
        Full markdown report as a single string.
    """
    lines: list[str] = []
    w = lines.append

    w(f"# {model_name} — Weight Outlier Analysis\n")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w("Tool: weight-outlier-analyzer\n")

    # ── Overview ──────────────────────────────────────────────────────────────
    w("## Overview\n")
    w(f"- **Tensors analyzed**: {summary['total_tensors']:,}")
    w(f"- **Parameters analyzed**: {format_number(summary['total_params_analyzed'])}")
    comp_str = ", ".join(
        f"{k}: {v}" for k, v in sorted(summary["component_counts"].items())
    )
    w(f"- **Components**: {comp_str}")
    if "global_abs_max" in summary:
        w(f"- **Global |max weight|**: {summary['global_abs_max']:.4f}")
    if "kurtosis_max" in summary:
        w(f"- **Peak kurtosis**: {summary['kurtosis_max']:.1f}")
    if "mean_clip_improvement_pct" in summary:
        w(f"- **Mean INT4 clip improvement**: {summary['mean_clip_improvement_pct']:.1f}%")
    w("")

    # ── Key findings ──────────────────────────────────────────────────────────
    w("## Key Findings\n")

    # Finding 1: overall distribution shape
    if "kurtosis_median" in summary:
        med_k = summary["kurtosis_median"]
        if med_k > 50:
            w(
                f"**Heavy-tailed distributions**: Median excess kurtosis is {med_k:.1f}, "
                f"far above the Gaussian baseline of 0. Weights have extreme outliers "
                f"that will damage naive quantization.\n"
            )
        elif med_k > 5:
            w(
                f"**Moderately heavy tails**: Median excess kurtosis is {med_k:.1f}, "
                f"indicating some outlier-prone tensors.\n"
            )
        else:
            w(
                f"**Near-Gaussian distributions**: Median kurtosis {med_k:.2f} suggests "
                f"relatively well-behaved weight distributions.\n"
            )

    # Finding 2: worst component
    if "component_stats" in summary:
        worst_comp = max(
            summary["component_stats"].items(),
            key=lambda x: x[1].get("max_kurtosis", 0),
        )
        w(
            f"**Worst component**: `{worst_comp[0]}` — max kurtosis "
            f"{worst_comp[1]['max_kurtosis']:.1f}, mean kurtosis "
            f"{worst_comp[1]['mean_kurtosis']:.1f} across "
            f"{worst_comp[1]['count']} tensors.\n"
        )

    # Finding 3: systematic outlier dimensions
    if "top_outlier_dims" in summary:
        top = summary["top_outlier_dims"][:5]
        dim_str = ", ".join(f"dim {d['dim']} ({d['pct']:.0f}%)" for d in top)
        w(
            f"**Systematic outlier dimensions**: {dim_str} — these dimensions are outliers "
            f"across many tensors, suggesting structural patterns in the weight space.\n"
        )

    # Finding 4: quantization impact
    if "max_clip_improvement_pct" in summary:
        w(
            f"**Quantization**: Clipping to 99.9th percentile before INT4 quantization "
            f"reduces MSE by up to {summary['max_clip_improvement_pct']:.0f}% "
            f"(mean {summary['mean_clip_improvement_pct']:.1f}%).\n"
        )

    # ── Visualizations ────────────────────────────────────────────────────────
    w("## Visualizations\n")

    chart_descriptions: dict[str, str] = {
        "Kurtosis by Layer": (
            "Excess kurtosis per tensor, grouped by layer and component. "
            "Higher values indicate heavier tails / more outliers."
        ),
        "Abs Max by Layer": (
            "Maximum absolute weight value per tensor. "
            "Spikes indicate tensors with extreme outlier values."
        ),
        "Dynamic Range": (
            "Ratio of max absolute value to median absolute value. "
            "Higher ratios mean worse quantization behavior."
        ),
        "Outlier Sigma Heatmap": (
            "Percentage of weights beyond Nσ thresholds, "
            "shown as a heatmap across layers and components."
        ),
        "Quantization Error": (
            "INT4 quantization mean squared error — comparing naive "
            "quantization vs 99.9th percentile clipping."
        ),
        "Outlier Dimensions": (
            "Input dimensions that are most frequently flagged as outliers. "
            "Systematic outlier dims affect many tensors."
        ),
        "Component Summary": (
            "Box-plot comparison of kurtosis, absolute max, and dynamic "
            "range across component types."
        ),
        "Worst Tensors": (
            "Top tensors ranked by excess kurtosis — the hardest to quantize."
        ),
        "Expert Heatmap (MoE)": (
            "For MoE models: kurtosis heatmap across (layer, expert). "
            "Reveals whether outlier behavior varies by expert."
        ),
    }

    for chart_name, filename in chart_manifest.items():
        desc = chart_descriptions.get(chart_name, "")
        w(f"### {chart_name}\n")
        if desc:
            w(f"{desc}\n")
        w(f"![[{images_subdir}/{filename}]]\n")

    # ── Component breakdown table ─────────────────────────────────────────────
    if "component_stats" in summary:
        w("## Component Breakdown\n")
        w("| Component | Count | Mean Kurtosis | Max Kurtosis | Mean |max| | Max |max| |")
        w("|-----------|------:|-------------:|------------:|----------:|--------:|")
        for comp, s in sorted(summary["component_stats"].items()):
            w(
                f"| {comp} | {s['count']} | {s['mean_kurtosis']:.1f} | "
                f"{s['max_kurtosis']:.1f} | {s['mean_abs_max']:.4f} | {s['max_abs_max']:.4f} |"
            )
        w("")

    # ── Worst tensors table ───────────────────────────────────────────────────
    if "worst_10" in summary:
        w("## Worst 10 Tensors (by Kurtosis)\n")
        w("| Tensor | Component | Kurtosis | |max| |")
        w("|--------|-----------|--------:|---------:|")
        for t in summary["worst_10"]:
            short_name = t["name"]
            if len(short_name) > 60:
                parts = short_name.split(".")
                short_name = "..." + ".".join(parts[-4:])
            w(
                f"| `{short_name}` | {t['component']} | "
                f"{t['kurtosis']:.1f} | {t['abs_max']:.4f} |"
            )
        w("")

    # ── Outlier dimensions table ──────────────────────────────────────────────
    if "top_outlier_dims" in summary:
        w("## Systematic Outlier Dimensions\n")
        w("| Dimension | Frequency | % of Tensors |")
        w("|----------:|----------:|------------:|")
        for d in summary["top_outlier_dims"]:
            w(f"| {d['dim']} | {d['freq']} | {d['pct']:.1f}% |")
        w("")

    # ── Sigma outlier density ─────────────────────────────────────────────────
    w("## Sigma Outlier Density\n")
    w("| Threshold | Mean % Beyond |")
    w("|----------:|--------------:|")
    for s in [3, 5, 8, 10]:
        key = f"mean_beyond_{s}sigma_pct"
        if key in summary:
            w(f"| {s}σ | {summary[key]:.4f}% |")
    w("")

    # ── Methodology ───────────────────────────────────────────────────────────
    w("## Methodology\n")
    w(
        "1. **Tensor selection**: Only 2D weight matrices (excluding embeddings, norms, "
        "biases, and routing gates)"
    )
    w(
        "2. **FP8 handling**: Models with FP8 weights are dequantized using "
        "`weight_scale_inv` block-wise scaling before analysis"
    )
    w("3. **Outlier detection**: Both sigma-based (3/5/8/10σ) and channel/dimension-based")
    w(
        "4. **Quantization simulation**: INT4 symmetric quantization with and without "
        "99.9th percentile clipping"
    )
    w(
        "5. **Kurtosis**: Excess kurtosis (Fisher definition, = 0 for Gaussian). "
        "Higher values indicate heavier tails"
    )
    w("")

    return "\n".join(lines)
