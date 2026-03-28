#!/usr/bin/env python3
"""
Step 2: Generate comprehensive visualizations from analyze.py JSONL output.

Produces charts covering ALL layers/tensors — no manual layer selection.
Charts are grouped into thematic sets and saved as PNGs.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

# matplotlib — use Agg backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ────────────────────────────────────────
# Load & parse
# ────────────────────────────────────────

def load_data(jsonl_path: str) -> list[dict]:
    """Load JSONL, skip error-only records."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "error" in rec and "std" not in rec:
                    continue  # skip failed tensors
                records.append(rec)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(records)} tensor records")
    return records


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_").lower()


# ────────────────────────────────────────
# Chart helpers
# ────────────────────────────────────────

STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#444",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "grid.alpha": 0.3,
    "font.size": 10,
}

COMPONENT_COLORS = {
    "attention": "#4fc3f7",
    "mlp_dense": "#ff8a65",
    "mlp_expert": "#ce93d8",
    "mlp_shared": "#81c784",
    "lm_head": "#fff176",
    "other": "#90a4ae",
}


def init_style():
    plt.rcParams.update(STYLE)


def save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def sorted_layers(records):
    """Return sorted unique layer indices."""
    idxs = set()
    for r in records:
        if r.get("layer_idx") is not None:
            idxs.add(r["layer_idx"])
    return sorted(idxs)


# ────────────────────────────────────────
# Chart generators
# ────────────────────────────────────────

def chart_kurtosis_by_layer(records, outdir, model_name):
    """Fig 1: Excess kurtosis per layer, colored by component type."""
    init_style()

    layers = sorted_layers(records)
    if not layers:
        print("  Skip kurtosis chart: no layer indices")
        return None

    fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.15), 6))

    by_comp = defaultdict(lambda: ([], []))
    for r in records:
        li = r.get("layer_idx")
        k = r.get("excess_kurtosis")
        if li is not None and k is not None:
            comp = r.get("component", "other")
            by_comp[comp][0].append(li)
            by_comp[comp][1].append(k)

    for comp, (xs, ys) in by_comp.items():
        ax.scatter(xs, ys, label=comp, alpha=0.5, s=12,
                   color=COMPONENT_COLORS.get(comp, "#aaa"))

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Excess Kurtosis")
    ax.set_title(f"{model_name} — Excess Kurtosis by Layer & Component")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_yscale("symlog", linthresh=10)
    ax.grid(True, alpha=0.2)

    path = os.path.join(outdir, "01_kurtosis_by_layer.png")
    save(fig, path)
    return path


def chart_abs_max_by_layer(records, outdir, model_name):
    """Fig 2: Absolute max value per layer, colored by component."""
    init_style()

    layers = sorted_layers(records)
    if not layers:
        return None

    fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.15), 6))

    by_comp = defaultdict(lambda: ([], []))
    for r in records:
        li = r.get("layer_idx")
        v = r.get("abs_max")
        if li is not None and v is not None:
            comp = r.get("component", "other")
            by_comp[comp][0].append(li)
            by_comp[comp][1].append(v)

    for comp, (xs, ys) in by_comp.items():
        ax.scatter(xs, ys, label=comp, alpha=0.5, s=12,
                   color=COMPONENT_COLORS.get(comp, "#aaa"))

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("|max weight|")
    ax.set_title(f"{model_name} — Absolute Max Weight by Layer")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)

    path = os.path.join(outdir, "02_abs_max_by_layer.png")
    save(fig, path)
    return path


def chart_dynamic_range(records, outdir, model_name):
    """Fig 3: Dynamic range (abs_max / median_abs) by layer."""
    init_style()

    layers = sorted_layers(records)
    if not layers:
        return None

    fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.15), 6))

    by_comp = defaultdict(lambda: ([], []))
    for r in records:
        li = r.get("layer_idx")
        dr = r.get("dynamic_range")
        if li is not None and dr is not None and dr < float("inf"):
            comp = r.get("component", "other")
            by_comp[comp][0].append(li)
            by_comp[comp][1].append(dr)

    for comp, (xs, ys) in by_comp.items():
        ax.scatter(xs, ys, label=comp, alpha=0.5, s=12,
                   color=COMPONENT_COLORS.get(comp, "#aaa"))

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Dynamic Range (abs_max / median_abs)")
    ax.set_title(f"{model_name} — Weight Dynamic Range by Layer")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)

    path = os.path.join(outdir, "03_dynamic_range_by_layer.png")
    save(fig, path)
    return path


def chart_outlier_sigma_heatmap(records, outdir, model_name):
    """Fig 4: Heatmap of beyond-Nsigma % across layers (aggregated by component)."""
    init_style()

    layers = sorted_layers(records)
    if not layers:
        return None

    components = ["attention", "mlp_dense", "mlp_expert", "mlp_shared"]
    sigma_levels = [3, 5, 8, 10]

    # For each (layer, component, sigma) compute mean beyond-Nsigma %
    agg = defaultdict(list)
    for r in records:
        li = r.get("layer_idx")
        comp = r.get("component", "other")
        if li is not None and comp in components:
            for s in sigma_levels:
                key = f"beyond_{s}sigma_pct"
                if key in r:
                    agg[(li, comp, s)].append(r[key])

    # Filter components that actually exist
    present_comps = [c for c in components if any(r.get("component") == c for r in records)]
    if not present_comps:
        return None

    n_rows = len(present_comps) * len(sigma_levels)
    row_labels = []
    data = np.zeros((n_rows, len(layers)))

    for ci, comp in enumerate(present_comps):
        for si, s in enumerate(sigma_levels):
            row_idx = ci * len(sigma_levels) + si
            row_labels.append(f"{comp} {s}σ")
            for li_idx, layer in enumerate(layers):
                vals = agg.get((layer, comp, s), [])
                data[row_idx, li_idx] = np.mean(vals) if vals else 0

    fig, ax = plt.subplots(figsize=(max(14, len(layers) * 0.12), max(5, n_rows * 0.35)))
    im = ax.imshow(data, aspect="auto", cmap="magma", interpolation="nearest")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)

    # Thin out x ticks for readability
    step = max(1, len(layers) // 40)
    ax.set_xticks(range(0, len(layers), step))
    ax.set_xticklabels([str(layers[i]) for i in range(0, len(layers), step)], fontsize=6, rotation=45)
    ax.set_xlabel("Layer Index")
    ax.set_title(f"{model_name} — Outlier Density (% beyond Nσ)")
    fig.colorbar(im, ax=ax, label="% of weights", shrink=0.8)

    path = os.path.join(outdir, "04_outlier_sigma_heatmap.png")
    save(fig, path, dpi=180)
    return path


def chart_quant_error(records, outdir, model_name):
    """Fig 5: INT4 quantization MSE — naive vs clipped, by layer."""
    init_style()

    layers = sorted_layers(records)
    has_quant = any("int4_naive_mse" in r for r in records)
    if not layers or not has_quant:
        return None

    # Aggregate by layer: mean improvement
    layer_improvement = defaultdict(list)
    layer_naive = defaultdict(list)
    layer_clip = defaultdict(list)
    for r in records:
        li = r.get("layer_idx")
        if li is None:
            continue
        if "int4_naive_mse" in r and "int4_clip999_mse" in r:
            layer_naive[li].append(r["int4_naive_mse"])
            layer_clip[li].append(r["int4_clip999_mse"])
            if "int4_clip_improvement_pct" in r:
                layer_improvement[li].append(r["int4_clip_improvement_pct"])

    xs = sorted(layer_naive.keys())
    naive_means = [np.mean(layer_naive[x]) for x in xs]
    clip_means = [np.mean(layer_clip[x]) for x in xs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(xs) * 0.15), 10))

    ax1.semilogy(xs, naive_means, label="Naive INT4", alpha=0.8, color="#ef5350", linewidth=1)
    ax1.semilogy(xs, clip_means, label="Clipped INT4 (0.1% clip)", alpha=0.8, color="#66bb6a", linewidth=1)
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Mean MSE (log)")
    ax1.set_title(f"{model_name} — INT4 Quantization Error by Layer")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    if layer_improvement:
        imp_means = [np.mean(layer_improvement[x]) for x in xs if x in layer_improvement]
        imp_xs = [x for x in xs if x in layer_improvement]
        ax2.bar(imp_xs, imp_means, alpha=0.7, color="#42a5f5", width=0.8)
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("MSE Improvement %")
        ax2.set_title("Clipping Improvement over Naive Quantization")
        ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(outdir, "05_quant_error_by_layer.png")
    save(fig, path)
    return path


def chart_outlier_dims(records, outdir, model_name):
    """Fig 6: Most frequent outlier input dimensions across all layers."""
    init_style()

    dim_freq = defaultdict(int)
    total_with_dims = 0
    for r in records:
        dims = r.get("input_dim_outlier_indices", [])
        if dims:
            total_with_dims += 1
            for d in dims:
                dim_freq[d] += 1

    if not dim_freq:
        return None

    # Top 30 dimensions by frequency
    top = sorted(dim_freq.items(), key=lambda x: -x[1])[:30]
    dims, counts = zip(*top)
    pcts = [c / total_with_dims * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(dims)), pcts, color="#42a5f5", alpha=0.8)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims], rotation=45, fontsize=8)
    ax.set_xlabel("Input Dimension Index")
    ax.set_ylabel("% of Tensors With This as Outlier Dim")
    ax.set_title(f"{model_name} — Most Frequent Outlier Input Dimensions")
    ax.grid(True, alpha=0.2, axis="y")

    path = os.path.join(outdir, "06_outlier_dims.png")
    save(fig, path)
    return path


def chart_component_summary(records, outdir, model_name):
    """Fig 7: Box-plot style summary comparing components."""
    init_style()

    comp_vals = defaultdict(lambda: {"kurtosis": [], "abs_max": [], "dynamic_range": []})
    for r in records:
        comp = r.get("component", "other")
        if r.get("excess_kurtosis") is not None:
            comp_vals[comp]["kurtosis"].append(r["excess_kurtosis"])
        if r.get("abs_max") is not None:
            comp_vals[comp]["abs_max"].append(r["abs_max"])
        if r.get("dynamic_range") is not None and r["dynamic_range"] < float("inf"):
            comp_vals[comp]["dynamic_range"].append(r["dynamic_range"])

    if not comp_vals:
        return None

    comps = sorted(comp_vals.keys())
    metrics = ["kurtosis", "abs_max", "dynamic_range"]
    titles = ["Excess Kurtosis", "Absolute Max", "Dynamic Range"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric, title in zip(axes, metrics, titles):
        data = [comp_vals[c][metric] for c in comps if comp_vals[c][metric]]
        labels = [c for c in comps if comp_vals[c][metric]]
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True,
                           boxprops=dict(facecolor="#42a5f5", alpha=0.4),
                           medianprops=dict(color="#ff8a65", linewidth=2))
            ax.set_title(title, fontsize=10)
            ax.tick_params(axis="x", rotation=30, labelsize=7)
            if metric in ("abs_max", "dynamic_range"):
                ax.set_yscale("symlog", linthresh=1)
            ax.grid(True, alpha=0.2)

    fig.suptitle(f"{model_name} — Component Comparison", fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, "07_component_summary.png")
    save(fig, path)
    return path


def chart_worst_tensors(records, outdir, model_name, top_n=30):
    """Fig 8: Top-N worst tensors by kurtosis — horizontal bar chart."""
    init_style()

    valid = [r for r in records if r.get("excess_kurtosis") is not None]
    if not valid:
        return None

    ranked = sorted(valid, key=lambda r: r["excess_kurtosis"], reverse=True)[:top_n]

    names = [r["tensor_name"].split(".")[-3] + "." + r["tensor_name"].split(".")[-2]
             if len(r["tensor_name"].split(".")) >= 3
             else r["tensor_name"][-30:]
             for r in ranked]
    vals = [r["excess_kurtosis"] for r in ranked]
    colors = [COMPONENT_COLORS.get(r.get("component", "other"), "#aaa") for r in ranked]

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.28)))
    ax.barh(range(len(names)), vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Excess Kurtosis")
    ax.set_title(f"{model_name} — Top {top_n} Worst Tensors by Kurtosis")
    ax.grid(True, alpha=0.2, axis="x")

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=comp) for comp, c in COMPONENT_COLORS.items()
                    if any(r.get("component") == comp for r in ranked)]
    if legend_elems:
        ax.legend(handles=legend_elems, fontsize=7, loc="lower right")

    path = os.path.join(outdir, "08_worst_tensors.png")
    save(fig, path)
    return path


def chart_expert_heatmap(records, outdir, model_name):
    """Fig 9: For MoE models — heatmap of kurtosis across (layer, expert)."""
    init_style()

    expert_recs = [r for r in records
                   if r.get("component") == "mlp_expert"
                   and r.get("layer_idx") is not None
                   and r.get("expert_idx") is not None]
    if len(expert_recs) < 10:
        print("  Skip expert heatmap: too few expert tensors")
        return None

    # Group by (layer, expert) — average kurtosis across proj types
    agg = defaultdict(list)
    for r in expert_recs:
        k = r.get("excess_kurtosis")
        if k is not None:
            agg[(r["layer_idx"], r["expert_idx"])].append(k)

    layers = sorted(set(k[0] for k in agg))
    experts = sorted(set(k[1] for k in agg))

    data = np.zeros((len(experts), len(layers)))
    for (li, ei), vals in agg.items():
        ri = experts.index(ei)
        ci = layers.index(li)
        data[ri, ci] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(14, len(layers) * 0.12), max(6, len(experts) * 0.08)))
    im = ax.imshow(data, aspect="auto", cmap="inferno", interpolation="nearest")

    step_x = max(1, len(layers) // 40)
    ax.set_xticks(range(0, len(layers), step_x))
    ax.set_xticklabels([str(layers[i]) for i in range(0, len(layers), step_x)], fontsize=6, rotation=45)

    step_y = max(1, len(experts) // 30)
    ax.set_yticks(range(0, len(experts), step_y))
    ax.set_yticklabels([str(experts[i]) for i in range(0, len(experts), step_y)], fontsize=6)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Expert Index")
    ax.set_title(f"{model_name} — Expert Kurtosis Heatmap")
    fig.colorbar(im, ax=ax, label="Mean Excess Kurtosis", shrink=0.8)

    path = os.path.join(outdir, "09_expert_heatmap.png")
    save(fig, path, dpi=180)
    return path


# ────────────────────────────────────────
# Main
# ────────────────────────────────────────

ALL_CHARTS = [
    ("Kurtosis by Layer", chart_kurtosis_by_layer),
    ("Abs Max by Layer", chart_abs_max_by_layer),
    ("Dynamic Range", chart_dynamic_range),
    ("Outlier Sigma Heatmap", chart_outlier_sigma_heatmap),
    ("Quantization Error", chart_quant_error),
    ("Outlier Dimensions", chart_outlier_dims),
    ("Component Summary", chart_component_summary),
    ("Worst Tensors", chart_worst_tensors),
    ("Expert Heatmap (MoE)", chart_expert_heatmap),
]


def run(jsonl_path: str, output_dir: str, model_name: str = None):
    os.makedirs(output_dir, exist_ok=True)
    records = load_data(jsonl_path)

    if not records:
        print("No records to visualize!")
        return []

    if model_name is None:
        model_name = os.path.basename(jsonl_path).replace("_stats.jsonl", "")

    generated = []
    for chart_name, chart_fn in ALL_CHARTS:
        print(f"Generating: {chart_name}")
        try:
            path = chart_fn(records, output_dir, model_name)
            if path:
                generated.append((chart_name, path))
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nGenerated {len(generated)}/{len(ALL_CHARTS)} charts in {output_dir}")

    # Save manifest
    manifest = {name: os.path.basename(path) for name, path in generated}
    manifest_path = os.path.join(output_dir, "chart_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return generated


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to JSONL from analyze.py")
    p.add_argument("--output-dir", required=True, help="Directory for chart PNGs")
    p.add_argument("--model-name", default=None, help="Model name for chart titles")
    args = p.parse_args()
    run(args.input, args.output_dir, args.model_name)
