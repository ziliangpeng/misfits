#!/usr/bin/env python3
"""
Custom visualization for Qwen3.5-0.8B — GPT-2-quality charts.

Qwen3.5-0.8B architecture:
- 24 layers, 1024 hidden dim
- Hybrid: 6 × (3 × Gated DeltaNet → FFN → 1 × Gated Attention → FFN)
- Components: linear_attn (DeltaNet), self_attn, mlp (gate_proj/up_proj/down_proj)
- Special: visual encoder, MTP head
"""

import json
import os
import gc
from collections import defaultdict

import numpy as np
import torch
from huggingface_hub import hf_hub_download, HfApi
from safetensors import safe_open

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

# ─── Style ────────────────────────────────────────
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

# Qwen3.5-specific component colors
COMP_COLORS = {
    "linear_attn": "#e74c3c",     # DeltaNet — red (the troublemakers)
    "self_attn":   "#4fc3f7",     # Standard attention — blue
    "mlp":         "#ff8a65",     # MLP — orange
    "visual":      "#ce93d8",     # Vision encoder — purple
    "mtp":         "#fff176",     # Multi-token prediction — yellow
    "other":       "#90a4ae",     # Misc
}

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_NAME = "Qwen3.5-0.8B"


def init_style():
    plt.rcParams.update(STYLE)


def save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Qwen3.5-specific classification ─────────────

def classify_qwen(name: str) -> dict:
    """Classify tensor into Qwen3.5-specific component types."""
    info = {"name": name, "layer_idx": None, "component": "other", "short_name": name}

    parts = name.split(".")

    # Extract layer index
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                info["layer_idx"] = int(parts[i + 1])
            except ValueError:
                pass
            break

    # Classify component
    if "linear_attn" in name:
        info["component"] = "linear_attn"
        # conv1d, in_proj, out_proj, etc.
        info["short_name"] = ".".join(parts[-2:]) if len(parts) >= 2 else name
    elif "self_attn" in name:
        info["component"] = "self_attn"
        info["short_name"] = ".".join(parts[-2:]) if len(parts) >= 2 else name
    elif "mlp" in name and "visual" not in name:
        info["component"] = "mlp"
        info["short_name"] = ".".join(parts[-2:]) if len(parts) >= 2 else name
    elif "visual" in name or "pos_embed" in name:
        info["component"] = "visual"
        # Shorten visual names
        if "blocks" in name:
            for i, p in enumerate(parts):
                if p == "blocks" and i + 1 < len(parts):
                    info["short_name"] = f"vis.b{parts[i+1]}.{'.'.join(parts[-2:])}"
                    break
        else:
            info["short_name"] = ".".join(parts[-3:]) if len(parts) >= 3 else name
    elif "mtp" in name:
        info["component"] = "mtp"
        info["short_name"] = ".".join(parts[-2:]) if len(parts) >= 2 else name

    return info


def should_analyze(name: str) -> bool:
    """Keep only 2D weight matrices relevant to quantization."""
    if "layernorm" in name or "rmsnorm" in name or "norm.weight" in name:
        return False
    if "embed_tokens" in name or "embed_positions" in name:
        return False
    if not name.endswith(".weight"):
        return False
    return True


# ─── Data loading ─────────────────────────────────

def load_model_tensors(model_id: str):
    """Download and iterate over all weight tensors."""
    api = HfApi()
    files = [f.rfilename for f in api.list_repo_tree(model_id) if hasattr(f, 'rfilename')]

    if "model.safetensors.index.json" in files:
        idx_path = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
    elif "model.safetensors" in files:
        p = hf_hub_download(model_id, "model.safetensors")
        with safe_open(p, framework="pt", device="cpu") as sf:
            weight_map = {k: "model.safetensors" for k in sf.keys()}
    else:
        raise FileNotFoundError(f"No safetensors in {model_id}")

    # Group by shard
    shard_names = defaultdict(list)
    for name, shard in weight_map.items():
        if should_analyze(name):
            shard_names[shard].append(name)

    tensors = {}
    for shard_file, names in sorted(shard_names.items()):
        shard_path = hf_hub_download(model_id, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for name in names:
                t = sf.get_tensor(name)
                if t.dim() >= 2:
                    tensors[name] = t.float()
        print(f"  Loaded {len(names)} tensors from {shard_file}")

    return tensors


def compute_stats(t: torch.Tensor) -> dict:
    flat = t.flatten()
    n = flat.numel()
    mean = flat.mean().item()
    std = flat.std().item()
    abs_max = flat.abs().max().item()
    median_abs = flat.abs().median().item()

    dev = (flat - mean).abs()
    outliers = {}
    for s in [3, 5, 8, 10]:
        c = (dev > s * std).sum().item()
        outliers[f"beyond_{s}sigma_count"] = int(c)
        outliers[f"beyond_{s}sigma_pct"] = c / n * 100

    kurtosis = ((flat - mean) / std).pow(4).mean().item() - 3.0 if std > 0 else 0.0
    dr = abs_max / median_abs if median_abs > 0 else float("inf")

    # INT4 quant error
    qerr = {}
    if abs_max > 0:
        scale = abs_max / 7.0
        q = torch.round(flat / scale).clamp(-8, 7) * scale
        mse_naive = (flat - q).pow(2).mean().item()
        clip_val = flat.abs().quantile(0.999).item()
        if clip_val > 0:
            sc = clip_val / 7.0
            qc = torch.round(flat.clamp(-clip_val, clip_val) / sc).clamp(-8, 7) * sc
            mse_clip = (flat - qc).pow(2).mean().item()
        else:
            mse_clip = mse_naive
        qerr = {
            "int4_naive_mse": mse_naive,
            "int4_clip999_mse": mse_clip,
            "int4_clip_improvement_pct": (mse_naive - mse_clip) / mse_naive * 100 if mse_naive > 0 else 0,
        }

    # Column (input dim) outliers
    col_outliers = []
    if t.dim() == 2 and t.shape[1] > 1:
        col_max = t.abs().amax(dim=0)
        col_med = col_max.median().item()
        col_outliers = (col_max > 3 * col_med).nonzero(as_tuple=True)[0].tolist()

    return {
        "shape": list(t.shape), "num_params": n,
        "mean": mean, "std": std, "abs_max": abs_max, "median_abs": median_abs,
        "dynamic_range": dr, "excess_kurtosis": kurtosis,
        "input_dim_outlier_indices": col_outliers[:30],
        **outliers, **qerr,
    }


# ─── Chart 1: Global Weight Distribution ─────────

def chart_global_distribution(tensors, outdir):
    """Fig 1: Global weight distribution with sigma lines (GPT-2 fig1 style)."""
    init_style()

    # Collect all weights
    all_weights = []
    total_params = 0
    for name, t in tensors.items():
        all_weights.append(t.flatten().numpy())
        total_params += t.numel()

    all_w = np.concatenate(all_weights)
    del all_weights

    mean = np.mean(all_w)
    std = np.std(all_w)
    abs_max = np.max(np.abs(all_w))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Full distribution histogram
    bins = np.linspace(np.min(all_w), np.max(all_w), 500)
    ax1.hist(all_w, bins=bins, density=True, color="#26c6da", alpha=0.8, edgecolor="none")

    for s, (color, style) in zip([3, 5, 10], [
        ("#ffa726", "--"), ("#ef5350", ":"), ("#e53935", "-.")
    ]):
        val = s * std
        ax1.axvline(mean + val, color=color, linestyle=style, linewidth=1.2,
                     label=f"{s} sigma ({val:.2f})")
        ax1.axvline(mean - val, color=color, linestyle=style, linewidth=1.2)

    ax1.set_xlabel("Weight Value")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Full Weight Distribution ({total_params/1e6:.0f}M params)")
    ax1.legend(fontsize=8)

    # Right: Absolute weight distribution on log scale
    abs_w = np.abs(all_w)
    bins_abs = np.linspace(0, np.max(abs_w), 500)
    ax2.hist(abs_w, bins=bins_abs, density=True, color="#26c6da", alpha=0.8, edgecolor="none")
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=1e-7)

    for s, (color, style) in zip([3, 5, 10], [
        ("#ffa726", "--"), ("#ef5350", ":"), ("#e53935", "-.")
    ]):
        val = s * std
        ax2.axvline(val, color=color, linestyle=style, linewidth=1.2,
                     label=f"{s} sigma = {val:.3f}")

    ax2.set_xlabel("|Weight Value|")
    ax2.set_ylabel("Density (log scale)")
    ax2.set_title("Absolute Weight Distribution (log scale)")
    ax2.legend(fontsize=8)

    fig.suptitle(f"{MODEL_NAME} Weight Distribution Overview", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(outdir, "fig1_global_distribution.png")
    save(fig, path, dpi=160)
    del all_w, abs_w
    return path


# ─── Chart 2: Per-Layer Stats (horizontal bars) ──

def chart_per_layer_stats(records, outdir):
    """Fig 2: Per-layer horizontal bars grouped by layer, showing max stats per component (GPT-2 fig2 style)."""
    init_style()

    # Group by (layer_idx, component) — take the worst tensor per group
    lang_recs = [r for r in records if r["layer_idx"] is not None
                 and r["component"] in ("linear_attn", "self_attn", "mlp")]

    # For each layer, show one row per component with the worst kurtosis tensor
    groups = defaultdict(list)
    for r in lang_recs:
        groups[(r["layer_idx"], r["component"])].append(r)

    rows = []
    for (lidx, comp), recs in sorted(groups.items()):
        worst = max(recs, key=lambda r: r["excess_kurtosis"])
        short = worst["short_name"].split(".")[-2] if "." in worst["short_name"] else worst["short_name"]
        rows.append({
            "label": f"L{lidx}.{comp}",
            "abs_max": max(r["abs_max"] for r in recs),
            "kurtosis": worst["excess_kurtosis"],
            "dynamic_range": max(r["dynamic_range"] for r in recs),
            "component": comp,
        })

    names = [r["label"] for r in rows]
    abs_maxes = [r["abs_max"] for r in rows]
    kurtoses = [r["kurtosis"] for r in rows]
    dyn_ranges = [r["dynamic_range"] for r in rows]
    colors = [COMP_COLORS.get(r["component"], "#aaa") for r in rows]

    n = len(names)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, max(8, n * 0.22)))

    y = range(n)

    ax1.barh(y, abs_maxes, color=colors, alpha=0.85, height=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=6)
    ax1.invert_yaxis()
    ax1.set_xlabel("Absolute Max Weight")
    ax1.set_title("Max |Weight| per Layer")

    ax2.barh(y, kurtoses, color=colors, alpha=0.85, height=0.8)
    ax2.set_yticks([])
    ax2.invert_yaxis()
    ax2.set_xscale("symlog", linthresh=1)
    ax2.set_xlabel("Excess Kurtosis (higher = heavier tails)")
    ax2.set_title("Weight Kurtosis")

    # Cap infinite dynamic ranges
    dr_capped = [min(d, 1000) for d in dyn_ranges]
    ax3.barh(y, dr_capped, color=colors, alpha=0.85, height=0.8)
    ax3.set_yticks([])
    ax3.invert_yaxis()
    ax3.set_xlabel("Dynamic Range (AbsMax / MedianAbs)")
    ax3.set_title("Dynamic Range per Layer")

    # Legend
    legend_elems = [Patch(facecolor=c, label=comp) for comp, c in COMP_COLORS.items()
                    if any(r["component"] == comp for r in rows)]
    ax1.legend(handles=legend_elems, fontsize=7, loc="lower right")

    fig.suptitle(f"{MODEL_NAME} Per-Layer Weight Statistics", fontsize=14, y=1.01)
    fig.tight_layout()

    path = os.path.join(outdir, "fig2_per_layer_stats.png")
    save(fig, path, dpi=160)
    return path


# ─── Chart 3: Worst Tensor Weight Heatmaps ───────

def chart_worst_heatmaps(tensors, records, outdir, top_n=4):
    """Fig 3: Actual weight matrix heatmaps for worst tensors (GPT-2 fig3 style)."""
    init_style()

    ranked = sorted(records, key=lambda r: r["excess_kurtosis"], reverse=True)

    # Pick top_n that are 2D, exist in tensors, and preferably from language model
    chosen = []
    # First pass: prefer language model tensors
    for r in ranked:
        if r["name"] in tensors and tensors[r["name"]].dim() == 2 and r["component"] in ("linear_attn", "self_attn", "mlp", "mtp"):
            chosen.append(r)
            if len(chosen) >= top_n:
                break
    # Fill remaining with any 2D tensor
    if len(chosen) < top_n:
        for r in ranked:
            if r["name"] in tensors and tensors[r["name"]].dim() == 2 and r not in chosen:
                chosen.append(r)
                if len(chosen) >= top_n:
                    break

    if not chosen:
        print("  Skip heatmap: no suitable tensors")
        return None

    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()

    for i, r in enumerate(chosen):
        ax = axes[i]
        t = tensors[r["name"]].abs().numpy()

        # Subsample for display
        max_display = 256
        h, w = t.shape
        step_h = max(1, h // max_display)
        step_w = max(1, w // max_display)
        t_sub = t[::step_h, ::step_w]

        im = ax.imshow(t_sub, aspect="auto", cmap="magma", interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8)

        short = r["name"].split("model.language_model.")[-1] if "model.language_model." in r["name"] else r["name"]
        ax.set_title(f"{short}\nKurtosis={r['excess_kurtosis']:.1f}  |  AbsMax={r['abs_max']:.2f}  |  Shape={r['shape']}",
                     fontsize=8)
        ax.set_xlabel("Output dim")
        ax.set_ylabel("Input dim")

    # Fill empty subplots
    for j in range(len(chosen), rows * cols):
        axes[j].axis("off")

    fig.suptitle(f"{MODEL_NAME} — Weight Heatmaps: Layers with Most Extreme Outliers", fontsize=13, y=1.01)
    fig.tight_layout()

    path = os.path.join(outdir, "fig3_worst_layers_heatmap.png")
    save(fig, path, dpi=160)
    return path


# ─── Chart 4: Systematic Outlier Dimensions ──────

def chart_outlier_dims(tensors, records, outdir):
    """Fig 4: 2-panel systematic outlier dimensions (GPT-2 fig4 style).
    Top: heatmap of per-input-dim max|weight| across DeltaNet conv1d layers.
    Bottom: average with labeled peaks."""
    init_style()

    # Use DeltaNet in_proj_qkv tensors — these are 2D [6144, 1024] with hidden_dim as input
    # They're the best equivalent to GPT-2's c_attn for outlier dim analysis
    target_suffix = "in_proj_qkv.weight"

    proj_tensors = {}
    for r in records:
        if r["component"] == "linear_attn" and r["name"].endswith(target_suffix):
            if r["name"] in tensors:
                proj_tensors[r["name"]] = (r["layer_idx"], tensors[r["name"]])

    # Fall back to self_attn q_proj if not enough
    if len(proj_tensors) < 3:
        for r in records:
            if r["component"] == "self_attn" and r["name"].endswith("q_proj.weight"):
                if r["name"] in tensors:
                    proj_tensors[r["name"]] = (r["layer_idx"], tensors[r["name"]])

    if len(proj_tensors) < 3:
        print("  Skip outlier dims: not enough suitable tensors")
        return None

    # Sort by layer
    items = sorted(proj_tensors.items(), key=lambda x: x[1][0])
    layer_labels = []
    dim_data = []

    for name, (lidx, t) in items:
        proj_type = name.split(".")[-2]  # e.g. in_proj_qkv or q_proj
        layer_labels.append(f"L{lidx} {proj_type}")
        # Per-input-dimension max |weight| (input dim = last dim = hidden_dim)
        if t.dim() == 2:
            col_max = t.abs().amax(dim=0).numpy()
        else:
            col_max = t.abs().flatten().numpy()
        dim_data.append(col_max)

    # Pad to same length
    max_dim = max(len(d) for d in dim_data)
    heatmap = np.zeros((len(dim_data), max_dim))
    for i, d in enumerate(dim_data):
        heatmap[i, :len(d)] = d

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [1.2, 1]})

    # Top: heatmap
    im = ax1.imshow(heatmap, aspect="auto", cmap="hot", interpolation="nearest")
    ax1.set_yticks(range(len(layer_labels)))
    ax1.set_yticklabels(layer_labels, fontsize=7)
    ax1.set_xlabel("Input Dimension Index")
    ax1.set_title(f"Per-Input-Dimension Max |Weight| Across DeltaNet in_proj_qkv Layers", fontsize=11)
    fig.colorbar(im, ax=ax1, shrink=0.6)

    # Bottom: average per-dim with outlier labels
    avg_per_dim = heatmap.mean(axis=0)
    median_val = np.median(avg_per_dim)

    bar_colors = np.where(avg_per_dim > 2 * median_val, "#ef5350", "#42a5f5")
    ax2.bar(range(max_dim), avg_per_dim, color=bar_colors, alpha=0.8, width=1.0)

    ax2.axhline(median_val, color="#66bb6a", linestyle="--", linewidth=1, label=f"Median = {median_val:.4f}")
    ax2.axhline(2 * median_val, color="#ffa726", linestyle="--", linewidth=1, label=f"2x Median = {2*median_val:.4f}")
    ax2.axhline(3 * median_val, color="#ef5350", linestyle="--", linewidth=1, label=f"3x Median = {3*median_val:.4f}")

    # Label top outlier dims
    top_indices = np.argsort(avg_per_dim)[-5:][::-1]
    for idx in top_indices:
        if avg_per_dim[idx] > 2 * median_val:
            ax2.annotate(f"dim {idx}", xy=(idx, avg_per_dim[idx]),
                        xytext=(idx + max_dim * 0.02, avg_per_dim[idx] * 1.05),
                        fontsize=7, color="#ef5350",
                        arrowprops=dict(arrowstyle="->", color="#ef5350", lw=0.8))

    ax2.set_xlabel("Input Dimension Index")
    ax2.set_ylabel("Avg Max |Weight| Across Layers")
    ax2.set_title(f"Average Per-Dimension Max |Weight| (Red = Outlier Dimensions)", fontsize=11)
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"{MODEL_NAME} — Systematic Outlier Dimensions in DeltaNet in_proj_qkv Weights", fontsize=13, y=1.01)
    fig.tight_layout()

    path = os.path.join(outdir, "fig4_systematic_outlier_dims.png")
    save(fig, path, dpi=160)
    return path


# ─── Chart 5: Quantization Impact ────────────────

def chart_quant_impact(tensors, records, outdir):
    """Fig 5: 3-panel quantization impact (GPT-2 fig5 style).
    Left: worst tensor distribution
    Middle: INT4 grid overlay
    Right: per-layer MSE comparison."""
    init_style()

    # Find worst tensor
    worst = max(records, key=lambda r: r["excess_kurtosis"])
    worst_name = worst["name"]
    worst_t = tensors.get(worst_name)

    if worst_t is None:
        # Find the worst that exists
        for r in sorted(records, key=lambda r: r["excess_kurtosis"], reverse=True):
            if r["name"] in tensors:
                worst = r
                worst_name = r["name"]
                worst_t = tensors[worst_name]
                break

    flat = worst_t.flatten().numpy()
    abs_max = np.max(np.abs(flat))
    std = np.std(flat)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Left: distribution of worst tensor
    bins = np.linspace(np.min(flat), np.max(flat), 400)
    ax1.hist(flat, bins=bins, density=True, color="#26c6da", alpha=0.8, edgecolor="none")

    for s, (color, style) in zip([3, 5, 10], [
        ("#ffa726", "--"), ("#ef5350", ":"), ("#e53935", "-.")
    ]):
        val = s * std
        ax1.axvline(val, color=color, linestyle=style, linewidth=1.2, label=f"{s} sigma")
        ax1.axvline(-val, color=color, linestyle=style, linewidth=1.2)

    short = worst_name.split("model.language_model.")[-1] if "model.language_model." in worst_name else worst_name
    ax1.set_title(f"{short}\nKurtosis={worst['excess_kurtosis']:.0f}, |Max|={abs_max:.2f}", fontsize=9)
    ax1.set_xlabel("Weight Value")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=7)

    # Middle: distribution with INT4 quantization levels
    clip_val = np.quantile(np.abs(flat), 0.999)

    # Zoom into the main mass
    xlim = max(abs(np.quantile(flat, 0.001)), abs(np.quantile(flat, 0.999))) * 1.5
    bins_zoom = np.linspace(-xlim, xlim, 200)
    ax2.hist(flat, bins=bins_zoom, density=True, color="#90a4ae", alpha=0.5, edgecolor="none",
             label="Weight distribution")

    # Naive INT4 levels
    naive_step = abs_max / 7.0
    naive_levels = np.array([i * naive_step for i in range(-8, 8)])
    for lvl in naive_levels:
        if -xlim <= lvl <= xlim:
            ax2.axvline(lvl, color="#ffa726", alpha=0.6, linewidth=0.5)
    ax2.axvline(naive_levels[0], color="#ffa726", alpha=0.6, linewidth=0.5, label=f"INT4 levels (no clip)\nstep={naive_step:.4f}")

    # Clipped INT4 levels
    clip_step = clip_val / 7.0
    clip_levels = np.array([i * clip_step for i in range(-8, 8)])
    for lvl in clip_levels:
        if -xlim <= lvl <= xlim:
            ax2.axvline(lvl, color="#66bb6a", alpha=0.7, linewidth=0.8)
    ax2.axvline(clip_levels[0], color="#66bb6a", alpha=0.7, linewidth=0.8,
                label=f"INT4 levels (99.9% clip)\nstep={clip_step:.4f}")

    finer = naive_step / clip_step if clip_step > 0 else 0
    ax2.set_title(f"INT4 Quantization Levels\nNo clip step={naive_step:.4f} vs Clip step={clip_step:.4f}\nClip = {finer:.0f}x finer", fontsize=9)
    ax2.set_xlabel("Weight Value")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=7)

    # Right: per-layer MSE comparison
    lang_recs = [r for r in records
                 if r["layer_idx"] is not None
                 and "int4_naive_mse" in r
                 and r["component"] in ("linear_attn", "self_attn", "mlp")]

    # Group by layer, take mean
    layer_naive = defaultdict(list)
    layer_clip = defaultdict(list)
    for r in lang_recs:
        layer_naive[r["layer_idx"]].append(r["int4_naive_mse"])
        layer_clip[r["layer_idx"]].append(r["int4_clip999_mse"])

    layers = sorted(layer_naive.keys())
    x = np.arange(len(layers))
    width = 0.35

    naive_means = [np.mean(layer_naive[l]) for l in layers]
    clip_means = [np.mean(layer_clip[l]) for l in layers]

    ax3.bar(x - width/2, naive_means, width, label="INT4 naive", color="#ef5350", alpha=0.8)
    ax3.bar(x + width/2, clip_means, width, label="INT4 + clip 99.9%", color="#66bb6a", alpha=0.8)

    ax3.set_xticks(x)
    ax3.set_xticklabels([str(l) for l in layers], fontsize=7, rotation=45)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("MSE (log scale)")
    ax3.set_yscale("log")
    ax3.set_title("INT4 Quantization Error per Layer")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.2)

    fig.suptitle(f"{MODEL_NAME} — Quantization Error: Impact of Outliers on INT4", fontsize=13, y=1.02)
    fig.tight_layout()

    path = os.path.join(outdir, "fig5_quantization_impact.png")
    save(fig, path, dpi=160)
    return path


# ─── Chart 6: DeltaNet conv1d Evolution by Layer ─

def chart_component_evolution(tensors, records, outdir):
    """Fig 6: Small-multiple distribution plots for the worst component type across layers
    (GPT-2 fig6 style — shows evolution by layer depth)."""
    init_style()

    # DeltaNet conv1d are the worst in Qwen3.5
    target_recs = sorted(
        [r for r in records if r["component"] == "linear_attn"
         and r["name"].endswith("conv1d.weight")
         and r["name"] in tensors
         and r["layer_idx"] is not None],
        key=lambda r: r["layer_idx"]
    )

    if len(target_recs) < 4:
        # Fall back to all linear_attn
        target_recs = sorted(
            [r for r in records if r["component"] == "linear_attn"
             and r["name"] in tensors
             and r["layer_idx"] is not None],
            key=lambda r: r["layer_idx"]
        )

    if len(target_recs) < 4:
        print("  Skip component evolution: not enough tensors")
        return None

    n = len(target_recs)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for i, r in enumerate(target_recs):
        ax = axes_flat[i]
        t = tensors[r["name"]].flatten().numpy()

        bins = np.linspace(np.min(t), np.max(t), 200)
        ax.hist(t, bins=bins, density=True, color="#42a5f5", alpha=0.8, edgecolor="none")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-5)

        std = np.std(t)
        for s_val in [3, 5]:
            ax.axvline(s_val * std, color="#ef5350", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvline(-s_val * std, color="#ef5350", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_title(f"Layer {r['layer_idx']}\nKurt={r['excess_kurtosis']:.1f}, |Max|={r['abs_max']:.2f}",
                     fontsize=8)
        ax.tick_params(labelsize=6)

    # Hide unused
    for j in range(len(target_recs), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"{MODEL_NAME} — DeltaNet conv1d: Weight Distribution by Layer Depth", fontsize=13, y=1.01)
    fig.tight_layout()

    path = os.path.join(outdir, "fig6_deltanet_conv1d_evolution.png")
    save(fig, path, dpi=150)
    return path


# ─── Main ─────────────────────────────────────────

def main():
    outdir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(outdir, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    tensors = load_model_tensors(MODEL_ID)
    print(f"Loaded {len(tensors)} tensors")

    # Compute stats for all tensors
    print("Computing statistics...")
    records = []
    for name, t in tensors.items():
        info = classify_qwen(name)
        stats = compute_stats(t)
        records.append({**info, **stats})
    print(f"Computed stats for {len(records)} tensors")

    # Generate charts
    print("\n=== Generating charts ===")

    print("Fig 1: Global Distribution...")
    chart_global_distribution(tensors, outdir)

    print("Fig 2: Per-Layer Stats...")
    chart_per_layer_stats(records, outdir)

    print("Fig 3: Worst Layer Heatmaps...")
    chart_worst_heatmaps(tensors, records, outdir)

    print("Fig 4: Systematic Outlier Dims...")
    chart_outlier_dims(tensors, records, outdir)

    print("Fig 5: Quantization Impact...")
    chart_quant_impact(tensors, records, outdir)

    print("Fig 6: DeltaNet conv1d Evolution...")
    chart_component_evolution(tensors, records, outdir)

    print("\n=== Done! ===")

    # Save records for the report
    records_path = os.path.join(os.path.dirname(__file__), "analysis_data.json")
    # Convert to JSON-serializable
    serializable = []
    for r in records:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (float, int, str, list, bool, type(None))):
                sr[k] = v
        serializable.append(sr)
    with open(records_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved analysis data to {records_path}")


if __name__ == "__main__":
    main()
