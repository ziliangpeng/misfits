#!/usr/bin/env python3
"""Weight outlier analysis for Llama-3.2-1B.

Llama-3.2-1B is a dense transformer with 16 layers, 2048 hidden dim, and
Grouped-Query Attention (32 Q heads, 8 KV heads). Despite being a modern
instruction-tuned model, its weight distributions are remarkably clean —
peak kurtosis is only ~16, making it an excellent baseline for comparison
with more exotic architectures.

Components:
- self_attn: q_proj (2048→2048), k_proj (2048→256), v_proj (2048→256),
             o_proj (2048→2048)
- mlp: gate_proj (2048→8192), up_proj (2048→8192), down_proj (8192→2048)
       (SwiGLU activation)

Key finding: gate_proj layers show a mild U-shaped kurtosis pattern
(early and late layers slightly worse than middle), but even the worst
tensor is well-behaved by GPT-2 standards.

Usage:
    python models/llama-3.2-1b/analyze.py
    python models/llama-3.2-1b/analyze.py --skip-stats   # reuse existing JSONL
"""

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Add repo root to path for shared imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.stats import compute_stats, should_analyze
from shared.io import get_weight_map, iter_tensors, load_tensor, load_jsonl, save_jsonl, load_done_set
from shared.viz import init_style, save_fig, DARK_STYLE
from shared.report import compute_summary, format_number, generate_markdown

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ─── Model config ────────────────────────────────
MODEL_ID   = "unsloth/Llama-3.2-1B"
MODEL_NAME = "Llama-3.2-1B"
MODEL_DIR  = Path(__file__).resolve().parent

# Llama component colors
COLORS = {
    "self_attn": "#4fc3f7",   # Attention — blue
    "mlp":       "#ff8a65",   # MLP — orange
    "other":     "#90a4ae",   # Misc (lm_head, etc.)
}


# ─── Llama-specific classification ───────────────

def classify_llama(name: str) -> dict:
    """Classify a Llama tensor into self_attn / mlp / other."""
    parts = name.split(".")
    info = {
        "tensor_name": name,
        "layer_idx":   None,
        "component":   "other",
        "proj_type":   parts[-2] if len(parts) >= 2 else "unknown",
        "expert_idx":  None,
    }

    # Llama uses 'model.layers.N.*' naming
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                info["layer_idx"] = int(parts[i + 1])
            except ValueError:
                pass
            break

    if "self_attn" in name:
        info["component"] = "self_attn"
    elif "mlp" in name:
        info["component"] = "mlp"

    return info


def should_analyze_llama(name: str) -> bool:
    """Llama-aware filter: skip norms, embeddings, and non-weight tensors."""
    if "layernorm" in name or "rmsnorm" in name or "norm.weight" in name:
        return False
    if "embed_tokens" in name or "embed_positions" in name:
        return False
    if not name.endswith(".weight"):
        return False
    return True


# ─── Step 1: Collect stats ───────────────────────

def collect_stats(skip: bool = False, token: str = None) -> list[dict]:
    """Collect per-tensor statistics, streaming one shard at a time.

    Writes results incrementally to stats.jsonl for full resume support.
    """
    jsonl_path = MODEL_DIR / "stats.jsonl"

    if skip and jsonl_path.exists():
        print(f"  Loading existing stats from {jsonl_path}")
        return load_jsonl(str(jsonl_path))

    print(f"  Fetching weight map for {MODEL_ID}...")
    weight_map = get_weight_map(MODEL_ID, token=token)

    targets = [n for n in weight_map if should_analyze_llama(n)]
    print(f"  {len(targets)} tensors to analyze")

    done      = load_done_set(str(jsonl_path))
    if done:
        print(f"  Resuming: {len(done)} already done")
    remaining = [n for n in targets if n not in done]

    if remaining:
        with open(jsonl_path, "a") as fh:
            for tname, t in iter_tensors(MODEL_ID, weight_map, remaining, token=token):
                if t.dim() < 2:
                    del t
                    continue
                t = t.float()
                meta   = classify_llama(tname)
                stats  = compute_stats(t)
                record = {**meta, **stats}
                fh.write(json.dumps(record) + "\n")
                del t
                gc.collect()

    return load_jsonl(str(jsonl_path))


# ─── Step 2: Model-specific visualizations ───────

def chart_global_distribution(weight_map: dict, records: list[dict],
                               images_dir: Path, token: str = None) -> str:
    """Fig 1: Streaming incremental histogram — full + log-scale absolute."""
    init_style()

    targets = [r["tensor_name"] for r in records if len(r.get("shape", [])) >= 2]

    # Pass 1: global min/max
    global_min, global_max = float("inf"), float("-inf")
    total_params = 0
    for tname, t in iter_tensors(MODEL_ID, weight_map, targets, token=token):
        t = t.float()
        total_params += t.numel()
        global_min = min(global_min, t.min().item())
        global_max = max(global_max, t.max().item())
        del t

    n_bins   = 500
    bins     = np.linspace(global_min, global_max, n_bins + 1)
    bins_abs = np.linspace(0, max(abs(global_min), abs(global_max)), n_bins + 1)
    hist_full = np.zeros(n_bins, dtype=np.float64)
    hist_abs  = np.zeros(n_bins, dtype=np.float64)
    sum_w, sum_w2 = 0.0, 0.0

    # Pass 2: accumulate
    for tname, t in iter_tensors(MODEL_ID, weight_map, targets, token=token):
        t    = t.float()
        flat = t.flatten().numpy()
        h, _   = np.histogram(flat, bins=bins)
        h_a, _ = np.histogram(np.abs(flat), bins=bins_abs)
        hist_full += h
        hist_abs  += h_a
        sum_w  += float(flat.sum())
        sum_w2 += float((flat ** 2).sum())
        del t, flat
        gc.collect()

    mean = sum_w / total_params
    std  = np.sqrt(sum_w2 / total_params - mean ** 2)

    density_full = hist_full / (total_params * np.diff(bins))
    density_abs  = hist_abs  / (total_params * np.diff(bins_abs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax1.fill_between(bin_centers, density_full, color="#26c6da", alpha=0.8)
    for s, (color, style) in zip([3, 5, 10],
            [("#ffa726", "--"), ("#ef5350", ":"), ("#e53935", "-.")]):
        val = s * std
        ax1.axvline(mean + val, color=color, linestyle=style, linewidth=1.2,
                    label=f"{s}σ ({val:.3f})")
        ax1.axvline(mean - val, color=color, linestyle=style, linewidth=1.2)
    ax1.set_xlabel("Weight Value"); ax1.set_ylabel("Density")
    ax1.set_title(f"Full Weight Distribution ({total_params/1e6:.0f}M params)")
    ax1.legend(fontsize=8)

    bin_centers_abs = (bins_abs[:-1] + bins_abs[1:]) / 2
    ax2.fill_between(bin_centers_abs, density_abs, color="#26c6da", alpha=0.8)
    ax2.set_yscale("log"); ax2.set_ylim(bottom=1e-7)
    for s, (color, style) in zip([3, 5, 10],
            [("#ffa726", "--"), ("#ef5350", ":"), ("#e53935", "-.")]):
        val = s * std
        ax2.axvline(val, color=color, linestyle=style, linewidth=1.2,
                    label=f"{s}σ = {val:.4f}")
    ax2.set_xlabel("|Weight Value|"); ax2.set_ylabel("Density (log scale)")
    ax2.set_title("Absolute Weight Distribution (log scale)")
    ax2.legend(fontsize=8)

    fig.suptitle(f"{MODEL_NAME} Weight Distribution Overview", fontsize=14, y=1.02)
    fig.tight_layout()

    path = str(images_dir / "fig1_global_distribution.png")
    save_fig(fig, path, dpi=160)
    return path


def chart_per_layer_stats(records: list[dict], images_dir: Path) -> str:
    """Fig 2: Horizontal bars — one row per (layer, component) showing worst tensor stats."""
    init_style()

    lang_recs = [r for r in records
                 if r.get("layer_idx") is not None
                 and r.get("component") in ("self_attn", "mlp")]

    groups = defaultdict(list)
    for r in lang_recs:
        groups[(r["layer_idx"], r["component"])].append(r)

    rows = []
    for (lidx, comp), recs in sorted(groups.items()):
        worst = max(recs, key=lambda r: r.get("excess_kurtosis", 0))
        rows.append({
            "label":         f"L{lidx}.{comp}",
            "abs_max":       max(r.get("abs_max", 0) for r in recs),
            "kurtosis":      worst.get("excess_kurtosis", 0),
            "dynamic_range": max((r.get("dynamic_range", 0) for r in recs
                                  if r.get("dynamic_range", float("inf")) < float("inf")),
                                 default=0.0),
            "component":     comp,
        })

    if not rows:
        return None

    names    = [r["label"]        for r in rows]
    abs_maxes = [r["abs_max"]       for r in rows]
    kurtoses  = [r["kurtosis"]      for r in rows]
    dr_capped = [min(r["dynamic_range"], 500) for r in rows]
    colors   = [COLORS.get(r["component"], "#aaa") for r in rows]
    n = len(names)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, max(8, n * 0.25)))
    y = range(n)

    ax1.barh(y, abs_maxes, color=colors, alpha=0.85, height=0.8)
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=7)
    ax1.invert_yaxis(); ax1.set_xlabel("Absolute Max Weight")
    ax1.set_title("Max |Weight| per Layer")

    ax2.barh(y, kurtoses, color=colors, alpha=0.85, height=0.8)
    ax2.set_yticks([]); ax2.invert_yaxis()
    ax2.set_xscale("symlog", linthresh=1)
    ax2.set_xlabel("Excess Kurtosis (higher = heavier tails)")
    ax2.set_title("Weight Kurtosis")

    ax3.barh(y, dr_capped, color=colors, alpha=0.85, height=0.8)
    ax3.set_yticks([]); ax3.invert_yaxis()
    ax3.set_xlabel("Dynamic Range (AbsMax / MedianAbs)")
    ax3.set_title("Dynamic Range per Layer")

    legend_elems = [Patch(facecolor=c, label=comp)
                    for comp, c in COLORS.items()
                    if any(r["component"] == comp for r in rows)]
    ax1.legend(handles=legend_elems, fontsize=7, loc="lower right")

    fig.suptitle(f"{MODEL_NAME} Per-Layer Weight Statistics", fontsize=14, y=1.01)
    fig.tight_layout()

    path = str(images_dir / "fig2_per_layer_stats.png")
    save_fig(fig, path, dpi=160)
    return path


def chart_worst_heatmaps(weight_map: dict, records: list[dict],
                         images_dir: Path, token: str = None, top_n: int = 4) -> str:
    """Fig 3: Heatmaps for the 4 tensors with highest kurtosis."""
    init_style()

    ranked = sorted(records, key=lambda r: r.get("excess_kurtosis", 0), reverse=True)
    chosen = [r for r in ranked
              if r.get("component") in ("self_attn", "mlp")
              and len(r.get("shape", [])) == 2][:top_n]

    if not chosen:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, r in enumerate(chosen):
        ax  = axes[i]
        t   = load_tensor(MODEL_ID, weight_map[r["tensor_name"]], r["tensor_name"], token=token)
        if t is None:
            ax.axis("off")
            continue

        t_abs = t.float().abs().numpy()
        del t

        max_display = 256
        h, w   = t_abs.shape
        step_h = max(1, h // max_display)
        step_w = max(1, w // max_display)
        t_sub  = t_abs[::step_h, ::step_w]
        del t_abs

        im = ax.imshow(t_sub, aspect="auto", cmap="magma", interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8)

        short = r["tensor_name"].replace("model.layers.", "L")
        ax.set_title(
            f"{short}\nKurtosis={r.get('excess_kurtosis', 0):.1f}  "
            f"|  AbsMax={r.get('abs_max', 0):.2f}  |  Shape={r.get('shape')}",
            fontsize=8)
        ax.set_xlabel("Output dim"); ax.set_ylabel("Input dim")
        del t_sub
        gc.collect()

    for j in range(len(chosen), 4):
        axes[j].axis("off")

    fig.suptitle(f"{MODEL_NAME} — Weight Heatmaps: Layers with Most Extreme Outliers",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    path = str(images_dir / "fig3_worst_layers_heatmap.png")
    save_fig(fig, path, dpi=160)
    return path


def chart_outlier_dims(weight_map: dict, records: list[dict],
                       images_dir: Path, token: str = None) -> str:
    """Fig 4: Systematic outlier input dimensions in q_proj weights across layers."""
    init_style()

    q_proj_recs = sorted(
        [r for r in records
         if r.get("component") == "self_attn"
         and r["tensor_name"].endswith("q_proj.weight")
         and r.get("layer_idx") is not None],
        key=lambda r: r["layer_idx"]
    )

    if len(q_proj_recs) < 3:
        print("  Skip outlier dims: not enough q_proj tensors")
        return None

    layer_labels = []
    dim_data     = []

    for r in q_proj_recs:
        t = load_tensor(MODEL_ID, weight_map[r["tensor_name"]], r["tensor_name"], token=token)
        if t is None or t.dim() != 2:
            continue
        t = t.float()
        layer_labels.append(f"L{r['layer_idx']}.q_proj")
        col_max = t.abs().amax(dim=0).numpy()
        dim_data.append(col_max)
        del t
        gc.collect()

    if not dim_data:
        return None

    max_dim = max(len(d) for d in dim_data)
    heatmap = np.zeros((len(dim_data), max_dim))
    for i, d in enumerate(dim_data):
        heatmap[i, :len(d)] = d

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={"height_ratios": [1.2, 1]})

    im = ax1.imshow(heatmap, aspect="auto", cmap="hot", interpolation="nearest")
    ax1.set_yticks(range(len(layer_labels)))
    ax1.set_yticklabels(layer_labels, fontsize=7)
    ax1.set_xlabel("Input Dimension Index")
    ax1.set_title("Per-Input-Dimension Max |Weight| Across q_proj Layers", fontsize=11)
    fig.colorbar(im, ax=ax1, shrink=0.6)

    avg_per_dim = heatmap.mean(axis=0)
    median_val  = np.median(avg_per_dim)

    bar_colors = np.where(avg_per_dim > 2 * median_val, "#ef5350", "#42a5f5")
    ax2.bar(range(max_dim), avg_per_dim, color=bar_colors, alpha=0.8, width=1.0)
    ax2.axhline(median_val,     color="#66bb6a", linestyle="--", linewidth=1,
                label=f"Median = {median_val:.4f}")
    ax2.axhline(2 * median_val, color="#ffa726", linestyle="--", linewidth=1,
                label=f"2× Median = {2*median_val:.4f}")
    ax2.axhline(3 * median_val, color="#ef5350", linestyle="--", linewidth=1,
                label=f"3× Median = {3*median_val:.4f}")

    top_indices = np.argsort(avg_per_dim)[-5:][::-1]
    for idx in top_indices:
        if avg_per_dim[idx] > 2 * median_val:
            ax2.annotate(
                f"dim {idx}", xy=(idx, avg_per_dim[idx]),
                xytext=(idx + max_dim * 0.02, avg_per_dim[idx] * 1.05),
                fontsize=7, color="#ef5350",
                arrowprops=dict(arrowstyle="->", color="#ef5350", lw=0.8))

    ax2.set_xlabel("Input Dimension Index")
    ax2.set_ylabel("Avg Max |Weight| Across Layers")
    ax2.set_title("Average Per-Dimension Max |Weight| (Red = Outlier Dimensions)", fontsize=11)
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"{MODEL_NAME} — Systematic Outlier Dimensions in q_proj Weights",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    path = str(images_dir / "fig4_systematic_outlier_dims.png")
    save_fig(fig, path, dpi=160)
    return path


def chart_quant_impact(weight_map: dict, records: list[dict],
                       images_dir: Path, token: str = None) -> str:
    """Fig 5: INT4 quantization impact — worst tensor distribution + per-layer MSE."""
    init_style()

    worst = next(
        (r for r in sorted(records, key=lambda r: r.get("excess_kurtosis", 0), reverse=True)
         if r.get("component") in ("self_attn", "mlp") and len(r.get("shape", [])) == 2),
        None)
    if worst is None:
        return None

    t    = load_tensor(MODEL_ID, weight_map[worst["tensor_name"]], worst["tensor_name"], token=token)
    flat = t.float().flatten().numpy()
    del t
    abs_max  = float(np.max(np.abs(flat)))
    std      = float(np.std(flat))
    clip_val = float(np.quantile(np.abs(flat), 0.999))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Left: worst tensor distribution
    bins = np.linspace(float(np.min(flat)), float(np.max(flat)), 400)
    ax1.hist(flat, bins=bins, density=True, color="#26c6da", alpha=0.8, edgecolor="none")
    for s, (color, style) in zip([3, 5, 10],
            [("#ffa726", "--"), ("#ef5350", ":"), ("#e53935", "-.")]):
        val = s * std
        ax1.axvline( val, color=color, linestyle=style, linewidth=1.2, label=f"{s}σ")
        ax1.axvline(-val, color=color, linestyle=style, linewidth=1.2)
    short = worst["tensor_name"].replace("model.layers.", "L")
    ax1.set_title(f"{short}\nKurtosis={worst.get('excess_kurtosis', 0):.1f}, "
                  f"|Max|={abs_max:.3f}", fontsize=9)
    ax1.set_xlabel("Weight Value"); ax1.set_ylabel("Density")
    ax1.legend(fontsize=7)

    # Middle: INT4 grid overlay
    xlim     = max(abs(np.quantile(flat, 0.001)), abs(np.quantile(flat, 0.999))) * 1.5
    bins_z   = np.linspace(-xlim, xlim, 200)
    ax2.hist(flat, bins=bins_z, density=True, color="#90a4ae", alpha=0.5, edgecolor="none",
             label="Weight distribution")
    naive_step = abs_max / 7.0
    for lvl in [i * naive_step for i in range(-8, 8)]:
        if -xlim <= lvl <= xlim:
            ax2.axvline(lvl, color="#ffa726", alpha=0.6, linewidth=0.5)
    ax2.axvline(-8 * naive_step, color="#ffa726", alpha=0.6, linewidth=0.5,
                label=f"INT4 (no clip)\nstep={naive_step:.4f}")
    clip_step = clip_val / 7.0
    for lvl in [i * clip_step for i in range(-8, 8)]:
        if -xlim <= lvl <= xlim:
            ax2.axvline(lvl, color="#66bb6a", alpha=0.7, linewidth=0.8)
    ax2.axvline(-8 * clip_step, color="#66bb6a", alpha=0.7, linewidth=0.8,
                label=f"INT4 (99.9% clip)\nstep={clip_step:.4f}")
    finer = naive_step / clip_step if clip_step > 0 else 0
    ax2.set_title(f"INT4 Quantization Levels\nNo clip step={naive_step:.4f} vs "
                  f"Clip step={clip_step:.4f}\nClip = {finer:.0f}× finer", fontsize=9)
    ax2.set_xlabel("Weight Value"); ax2.set_ylabel("Density")
    ax2.legend(fontsize=7)
    del flat; gc.collect()

    # Right: per-layer MSE comparison
    lang_recs   = [r for r in records
                   if r.get("layer_idx") is not None
                   and "int4_naive_mse" in r
                   and r.get("component") in ("self_attn", "mlp")]
    layer_naive = defaultdict(list)
    layer_clip  = defaultdict(list)
    for r in lang_recs:
        layer_naive[r["layer_idx"]].append(r["int4_naive_mse"])
        layer_clip [r["layer_idx"]].append(r["int4_clip999_mse"])

    layers = sorted(layer_naive.keys())
    x      = np.arange(len(layers))
    width  = 0.35
    ax3.bar(x - width/2, [np.mean(layer_naive[l]) for l in layers],
            width, label="INT4 naive",        color="#ef5350", alpha=0.8)
    ax3.bar(x + width/2, [np.mean(layer_clip[l])  for l in layers],
            width, label="INT4 + clip 99.9%", color="#66bb6a", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(l) for l in layers], fontsize=7)
    ax3.set_xlabel("Layer"); ax3.set_ylabel("MSE (log scale)")
    ax3.set_yscale("log"); ax3.set_title("INT4 Quantization Error per Layer")
    ax3.legend(fontsize=7); ax3.grid(True, alpha=0.2)

    fig.suptitle(f"{MODEL_NAME} — Quantization Error: Impact of Outliers on INT4",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    path = str(images_dir / "fig5_quantization_impact.png")
    save_fig(fig, path, dpi=160)
    return path


def chart_component_evolution(weight_map: dict, records: list[dict],
                               images_dir: Path, token: str = None) -> str:
    """Fig 6: MLP gate_proj distribution by layer depth — SwiGLU gating weights."""
    init_style()

    target_recs = sorted(
        [r for r in records
         if r.get("component") == "mlp"
         and r["tensor_name"].endswith("gate_proj.weight")
         and r.get("layer_idx") is not None],
        key=lambda r: r["layer_idx"]
    )

    if len(target_recs) < 4:
        # Fall back to all mlp tensors
        target_recs = sorted(
            [r for r in records
             if r.get("component") == "mlp" and r.get("layer_idx") is not None],
            key=lambda r: r["layer_idx"]
        )

    if len(target_recs) < 2:
        print("  Skip component evolution: not enough tensors")
        return None

    n    = len(target_recs)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for i, r in enumerate(target_recs):
        ax = axes_flat[i]
        t  = load_tensor(MODEL_ID, weight_map[r["tensor_name"]], r["tensor_name"], token=token)
        if t is None:
            ax.axis("off")
            continue

        flat = t.float().flatten().numpy()
        del t

        bins = np.linspace(float(np.min(flat)), float(np.max(flat)), 200)
        ax.hist(flat, bins=bins, density=True, color="#42a5f5", alpha=0.8, edgecolor="none")
        ax.set_yscale("log"); ax.set_ylim(bottom=1e-5)

        std = float(np.std(flat))
        for s_val in [3, 5]:
            ax.axvline( s_val * std, color="#ef5350", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvline(-s_val * std, color="#ef5350", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_title(
            f"Layer {r['layer_idx']}\nKurt={r.get('excess_kurtosis', 0):.1f}, "
            f"|Max|={r.get('abs_max', 0):.3f}",
            fontsize=8)
        ax.tick_params(labelsize=6)
        del flat
        gc.collect()

    for j in range(len(target_recs), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"{MODEL_NAME} — MLP gate_proj: Weight Distribution by Layer Depth",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    path = str(images_dir / "fig6_mlp_gate_proj_evolution.png")
    save_fig(fig, path, dpi=150)
    return path


def generate_charts(records: list[dict], token: str = None) -> dict:
    """Generate all 6 charts. Returns manifest mapping display name → filename."""
    images_dir = MODEL_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    print("  Fetching weight map for charts...")
    weight_map = get_weight_map(MODEL_ID, token=token)

    manifest = {}

    print("  Fig 1: Global Distribution...")
    p = chart_global_distribution(weight_map, records, images_dir, token=token)
    if p: manifest["Global Distribution"] = Path(p).name
    gc.collect()

    print("  Fig 2: Per-Layer Stats...")
    p = chart_per_layer_stats(records, images_dir)
    if p: manifest["Per-Layer Statistics"] = Path(p).name
    gc.collect()

    print("  Fig 3: Worst Tensor Heatmaps...")
    p = chart_worst_heatmaps(weight_map, records, images_dir, token=token)
    if p: manifest["Worst Tensor Heatmaps"] = Path(p).name
    gc.collect()

    print("  Fig 4: Systematic Outlier Dims...")
    p = chart_outlier_dims(weight_map, records, images_dir, token=token)
    if p: manifest["Systematic Outlier Dimensions"] = Path(p).name
    gc.collect()

    print("  Fig 5: Quantization Impact...")
    p = chart_quant_impact(weight_map, records, images_dir, token=token)
    if p: manifest["Quantization Impact"] = Path(p).name
    gc.collect()

    print("  Fig 6: MLP gate_proj Evolution...")
    p = chart_component_evolution(weight_map, records, images_dir, token=token)
    if p: manifest["MLP gate_proj Evolution"] = Path(p).name
    gc.collect()

    # Save manifest
    manifest_path = images_dir / "chart_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest


# ─── Step 3: Report ──────────────────────────────

def write_report(records: list[dict], chart_manifest: dict) -> None:
    """Generate markdown analysis report and summary JSON."""
    summary = compute_summary(records)
    md      = generate_markdown(summary, MODEL_NAME, "images", chart_manifest)

    report_path = MODEL_DIR / "llama_3_2_1b_weight_outlier_analysis.md"
    report_path.write_text(md)
    print(f"  Report: {report_path}")

    summary_path = MODEL_DIR / "llama_3_2_1b_weight_outlier_analysis_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"  Summary JSON: {summary_path}")


# ─── Main ────────────────────────────────────────

def main(skip_stats: bool = False, token: str = None):
    """Run full analysis pipeline for Llama-3.2-1B."""
    print("=" * 60)
    print(f"  {MODEL_NAME} — Weight Outlier Analysis")
    print("=" * 60)

    t0 = time.time()

    print("\n[1/3] Collecting stats...")
    records = collect_stats(skip=skip_stats, token=token)
    print(f"  Stats: {len(records)} tensors")

    print("\n[2/3] Generating charts...")
    chart_manifest = generate_charts(records, token=token)
    print(f"  Charts: {len(chart_manifest)} generated")

    print("\n[3/3] Writing report...")
    write_report(records, chart_manifest)

    print(f"\nDone in {time.time() - t0:.0f}s")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weight outlier analysis for Llama-3.2-1B")
    parser.add_argument("--skip-stats", action="store_true",
                        help="Skip stats collection and reuse existing stats.jsonl")
    parser.add_argument("--token", default=None,
                        help="HuggingFace access token (for gated models)")
    args = parser.parse_args()
    main(skip_stats=args.skip_stats, token=args.token)
