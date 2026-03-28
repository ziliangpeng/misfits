#!/usr/bin/env python3
"""
Custom visualization for Llama-3.2-1B — GPT-2-quality charts.

Llama-3.2-1B architecture:
- 16 layers, 2048 hidden dim
- Standard transformer: self_attn (GQA: 32 Q heads, 8 KV heads) + SwiGLU MLP
- Components: self_attn (q/k/v/o_proj), mlp (gate/up/down_proj)

Memory-optimized: processes tensors one shard at a time.
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

# Llama component colors
COMP_COLORS = {
    "self_attn":  "#4fc3f7",     # Attention — blue
    "mlp":        "#ff8a65",     # MLP — orange
    "other":      "#90a4ae",     # Misc
}

MODEL_ID = "unsloth/Llama-3.2-1B"
MODEL_NAME = "Llama-3.2-1B"


def init_style():
    plt.rcParams.update(STYLE)


def save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Classification ───────────────────────────────

def classify_llama(name: str) -> dict:
    info = {"name": name, "layer_idx": None, "component": "other", "short_name": name}
    parts = name.split(".")

    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                info["layer_idx"] = int(parts[i + 1])
            except ValueError:
                pass
            break

    if "self_attn" in name:
        info["component"] = "self_attn"
        info["short_name"] = parts[-2] + "." + parts[-1] if len(parts) >= 2 else name
    elif "mlp" in name:
        info["component"] = "mlp"
        info["short_name"] = parts[-2] + "." + parts[-1] if len(parts) >= 2 else name

    return info


def should_analyze(name: str) -> bool:
    if "layernorm" in name or "rmsnorm" in name or "norm.weight" in name:
        return False
    if "embed_tokens" in name or "embed_positions" in name:
        return False
    if not name.endswith(".weight"):
        return False
    return True


# ─── Data loading ─────────────────────────────────

def get_shard_info(model_id: str):
    """Get weight map without loading tensors."""
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

    shard_names = defaultdict(list)
    for name, shard in weight_map.items():
        if should_analyze(name):
            shard_names[shard].append(name)

    return shard_names


def load_tensor(model_id: str, shard_file: str, name: str):
    """Load a single tensor."""
    shard_path = hf_hub_download(model_id, shard_file)
    with safe_open(shard_path, framework="pt", device="cpu") as sf:
        t = sf.get_tensor(name)
        if t.dim() >= 2:
            return t.float()
    return None


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


# ─── Phase 1: Compute all stats (memory-light) ───

def compute_all_stats(model_id, shard_names):
    """Compute stats tensor-by-tensor, releasing each after stats are computed."""
    records = []
    for shard_file, names in sorted(shard_names.items()):
        shard_path = hf_hub_download(model_id, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for name in names:
                t = sf.get_tensor(name)
                if t.dim() >= 2:
                    t = t.float()
                    info = classify_llama(name)
                    stats = compute_stats(t)
                    records.append({**info, **stats})
                    del t
        gc.collect()
        print(f"  Stats from {shard_file}: {len(names)} tensors")
    return records


# ─── Chart 1: Global Weight Distribution ─────────

def chart_global_distribution(model_id, shard_names, outdir):
    """Fig 1: Build histogram incrementally to avoid OOM."""
    init_style()

    # Use numpy histograms incrementally
    # First pass: find global min/max
    global_min, global_max = float("inf"), float("-inf")
    total_params = 0

    for shard_file, names in sorted(shard_names.items()):
        shard_path = hf_hub_download(model_id, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for name in names:
                t = sf.get_tensor(name)
                if t.dim() >= 2:
                    t = t.float()
                    total_params += t.numel()
                    tmin, tmax = t.min().item(), t.max().item()
                    global_min = min(global_min, tmin)
                    global_max = max(global_max, tmax)
                    del t
        gc.collect()

    n_bins = 500
    bins = np.linspace(global_min, global_max, n_bins + 1)
    bins_abs = np.linspace(0, max(abs(global_min), abs(global_max)), n_bins + 1)
    hist_full = np.zeros(n_bins, dtype=np.float64)
    hist_abs = np.zeros(n_bins, dtype=np.float64)

    # Accumulate running stats for mean/std
    sum_w = 0.0
    sum_w2 = 0.0

    for shard_file, names in sorted(shard_names.items()):
        shard_path = hf_hub_download(model_id, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for name in names:
                t = sf.get_tensor(name)
                if t.dim() >= 2:
                    t = t.float()
                    flat = t.flatten().numpy()
                    h, _ = np.histogram(flat, bins=bins)
                    hist_full += h
                    h_abs, _ = np.histogram(np.abs(flat), bins=bins_abs)
                    hist_abs += h_abs
                    sum_w += flat.sum()
                    sum_w2 += (flat ** 2).sum()
                    del t, flat
        gc.collect()

    mean = sum_w / total_params
    std = np.sqrt(sum_w2 / total_params - mean ** 2)

    # Convert to density
    bin_widths = np.diff(bins)
    density_full = hist_full / (total_params * bin_widths)
    bin_widths_abs = np.diff(bins_abs)
    density_abs = hist_abs / (total_params * bin_widths_abs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax1.fill_between(bin_centers, density_full, color="#26c6da", alpha=0.8)

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

    bin_centers_abs = (bins_abs[:-1] + bins_abs[1:]) / 2
    ax2.fill_between(bin_centers_abs, density_abs, color="#26c6da", alpha=0.8)
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
    return path, mean, std


# ─── Chart 2: Per-Layer Stats ────────────────────

def chart_per_layer_stats(records, outdir):
    """Fig 2: Per-layer horizontal bars — one row per (layer, component) showing worst tensor."""
    init_style()

    lang_recs = [r for r in records if r["layer_idx"] is not None
                 and r["component"] in ("self_attn", "mlp")]

    groups = defaultdict(list)
    for r in lang_recs:
        groups[(r["layer_idx"], r["component"])].append(r)

    rows = []
    for (lidx, comp), recs in sorted(groups.items()):
        worst = max(recs, key=lambda r: r["excess_kurtosis"])
        proj = worst["short_name"].split(".")[-2] if "." in worst["short_name"] else ""
        rows.append({
            "label": f"h.{lidx}.{comp}",
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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, max(8, n * 0.25)))

    y = range(n)

    ax1.barh(y, abs_maxes, color=colors, alpha=0.85, height=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel("Absolute Max Weight")
    ax1.set_title("Max |Weight| per Layer")

    ax2.barh(y, kurtoses, color=colors, alpha=0.85, height=0.8)
    ax2.set_yticks([])
    ax2.invert_yaxis()
    ax2.set_xscale("symlog", linthresh=1)
    ax2.set_xlabel("Excess Kurtosis (higher = heavier tails)")
    ax2.set_title("Weight Kurtosis")

    dr_capped = [min(d, 500) for d in dyn_ranges]
    ax3.barh(y, dr_capped, color=colors, alpha=0.85, height=0.8)
    ax3.set_yticks([])
    ax3.invert_yaxis()
    ax3.set_xlabel("Dynamic Range (AbsMax / MedianAbs)")
    ax3.set_title("Dynamic Range per Layer")

    legend_elems = [Patch(facecolor=c, label=comp) for comp, c in COMP_COLORS.items()
                    if any(r["component"] == comp for r in rows)]
    ax1.legend(handles=legend_elems, fontsize=7, loc="lower right")

    fig.suptitle(f"{MODEL_NAME} Per-Layer Weight Statistics", fontsize=14, y=1.01)
    fig.tight_layout()

    path = os.path.join(outdir, "fig2_per_layer_stats.png")
    save(fig, path, dpi=160)
    return path


# ─── Chart 3: Worst Tensor Heatmaps ─────────────

def chart_worst_heatmaps(model_id, shard_names, records, outdir, top_n=4):
    """Fig 3: Load only the 4 worst tensors for heatmaps."""
    init_style()

    ranked = sorted(records, key=lambda r: r["excess_kurtosis"], reverse=True)

    # Pick top_n 2D tensors
    chosen = []
    for r in ranked:
        if r["component"] in ("self_attn", "mlp") and len(r["shape"]) == 2:
            chosen.append(r)
            if len(chosen) >= top_n:
                break

    if not chosen:
        return None

    # Build name→shard map
    name_to_shard = {}
    for shard_file, names in shard_names.items():
        for name in names:
            name_to_shard[name] = shard_file

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, r in enumerate(chosen):
        ax = axes[i]
        t = load_tensor(model_id, name_to_shard[r["name"]], r["name"])
        if t is None:
            ax.axis("off")
            continue

        t_abs = t.abs().numpy()
        del t

        max_display = 256
        h, w = t_abs.shape
        step_h = max(1, h // max_display)
        step_w = max(1, w // max_display)
        t_sub = t_abs[::step_h, ::step_w]
        del t_abs

        im = ax.imshow(t_sub, aspect="auto", cmap="magma", interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8)

        short = r["name"].replace("model.layers.", "h.")
        ax.set_title(f"{short}\nKurtosis={r['excess_kurtosis']:.1f}  |  AbsMax={r['abs_max']:.2f}  |  Shape={r['shape']}",
                     fontsize=8)
        ax.set_xlabel("Output dim")
        ax.set_ylabel("Input dim")
        del t_sub
        gc.collect()

    for j in range(len(chosen), 4):
        axes[j].axis("off")

    fig.suptitle(f"{MODEL_NAME} — Weight Heatmaps: Layers with Most Extreme Outliers", fontsize=13, y=1.01)
    fig.tight_layout()

    path = os.path.join(outdir, "fig3_worst_layers_heatmap.png")
    save(fig, path, dpi=160)
    return path


# ─── Chart 4: Systematic Outlier Dims ────────────

def chart_outlier_dims(model_id, shard_names, records, outdir):
    """Fig 4: 2-panel systematic outlier dims using self_attn q_proj."""
    init_style()

    # Find q_proj records
    q_proj_recs = sorted(
        [r for r in records if r["component"] == "self_attn"
         and r["name"].endswith("q_proj.weight")
         and r["layer_idx"] is not None],
        key=lambda r: r["layer_idx"]
    )

    if len(q_proj_recs) < 3:
        print("  Skip outlier dims: not enough q_proj tensors")
        return None

    # Build name→shard map
    name_to_shard = {}
    for shard_file, names in shard_names.items():
        for name in names:
            name_to_shard[name] = shard_file

    layer_labels = []
    dim_data = []

    for r in q_proj_recs:
        t = load_tensor(model_id, name_to_shard[r["name"]], r["name"])
        if t is None or t.dim() != 2:
            continue
        layer_labels.append(f"h.{r['layer_idx']}.q_proj")
        col_max = t.abs().amax(dim=0).numpy()
        dim_data.append(col_max)
        del t
        gc.collect()

    max_dim = max(len(d) for d in dim_data)
    heatmap = np.zeros((len(dim_data), max_dim))
    for i, d in enumerate(dim_data):
        heatmap[i, :len(d)] = d

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [1.2, 1]})

    im = ax1.imshow(heatmap, aspect="auto", cmap="hot", interpolation="nearest")
    ax1.set_yticks(range(len(layer_labels)))
    ax1.set_yticklabels(layer_labels, fontsize=7)
    ax1.set_xlabel("Input Dimension Index")
    ax1.set_title(f"Per-Input-Dimension Max |Weight| Across q_proj Layers", fontsize=11)
    fig.colorbar(im, ax=ax1, shrink=0.6)

    avg_per_dim = heatmap.mean(axis=0)
    median_val = np.median(avg_per_dim)

    bar_colors = np.where(avg_per_dim > 2 * median_val, "#ef5350", "#42a5f5")
    ax2.bar(range(max_dim), avg_per_dim, color=bar_colors, alpha=0.8, width=1.0)

    ax2.axhline(median_val, color="#66bb6a", linestyle="--", linewidth=1, label=f"Median = {median_val:.4f}")
    ax2.axhline(2 * median_val, color="#ffa726", linestyle="--", linewidth=1, label=f"2x Median = {2*median_val:.4f}")
    ax2.axhline(3 * median_val, color="#ef5350", linestyle="--", linewidth=1, label=f"3x Median = {3*median_val:.4f}")

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

    fig.suptitle(f"{MODEL_NAME} — Systematic Outlier Dimensions in q_proj Weights", fontsize=13, y=1.01)
    fig.tight_layout()

    path = os.path.join(outdir, "fig4_systematic_outlier_dims.png")
    save(fig, path, dpi=160)
    return path


# ─── Chart 5: Quantization Impact ────────────────

def chart_quant_impact(model_id, shard_names, records, outdir):
    init_style()

    # Find worst tensor
    worst = None
    for r in sorted(records, key=lambda r: r["excess_kurtosis"], reverse=True):
        if r["component"] in ("self_attn", "mlp") and len(r["shape"]) == 2:
            worst = r
            break

    if worst is None:
        return None

    # Build name→shard map
    name_to_shard = {}
    for shard_file, names in shard_names.items():
        for name in names:
            name_to_shard[name] = shard_file

    t = load_tensor(model_id, name_to_shard[worst["name"]], worst["name"])
    flat = t.flatten().numpy()
    del t
    abs_max = np.max(np.abs(flat))
    std = np.std(flat)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Left: distribution
    bins = np.linspace(np.min(flat), np.max(flat), 400)
    ax1.hist(flat, bins=bins, density=True, color="#26c6da", alpha=0.8, edgecolor="none")

    for s, (color, style) in zip([3, 5, 10], [
        ("#ffa726", "--"), ("#ef5350", ":"), ("#e53935", "-.")
    ]):
        val = s * std
        ax1.axvline(val, color=color, linestyle=style, linewidth=1.2, label=f"{s} sigma")
        ax1.axvline(-val, color=color, linestyle=style, linewidth=1.2)

    short = worst["name"].replace("model.layers.", "h.")
    ax1.set_title(f"{short}\nKurtosis={worst['excess_kurtosis']:.1f}, |Max|={abs_max:.2f}", fontsize=9)
    ax1.set_xlabel("Weight Value")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=7)

    # Middle: INT4 grid overlay
    clip_val = np.quantile(np.abs(flat), 0.999)
    xlim = max(abs(np.quantile(flat, 0.001)), abs(np.quantile(flat, 0.999))) * 1.5
    bins_zoom = np.linspace(-xlim, xlim, 200)
    ax2.hist(flat, bins=bins_zoom, density=True, color="#90a4ae", alpha=0.5, edgecolor="none",
             label="Weight distribution")

    naive_step = abs_max / 7.0
    naive_levels = np.array([i * naive_step for i in range(-8, 8)])
    for lvl in naive_levels:
        if -xlim <= lvl <= xlim:
            ax2.axvline(lvl, color="#ffa726", alpha=0.6, linewidth=0.5)
    ax2.axvline(naive_levels[0], color="#ffa726", alpha=0.6, linewidth=0.5,
                label=f"INT4 levels (no clip)\nstep={naive_step:.4f}")

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

    del flat
    gc.collect()

    # Right: per-layer MSE
    lang_recs = [r for r in records
                 if r["layer_idx"] is not None
                 and "int4_naive_mse" in r
                 and r["component"] in ("self_attn", "mlp")]

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
    ax3.set_xticklabels([str(l) for l in layers], fontsize=7)
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


# ─── Chart 6: MLP gate_proj Evolution ────────────

def chart_component_evolution(model_id, shard_names, records, outdir):
    """Fig 6: Small multiples of MLP gate_proj across all layers."""
    init_style()

    target_recs = sorted(
        [r for r in records if r["component"] == "mlp"
         and r["name"].endswith("gate_proj.weight")
         and r["layer_idx"] is not None],
        key=lambda r: r["layer_idx"]
    )

    if len(target_recs) < 4:
        print("  Skip component evolution: not enough tensors")
        return None

    # Build name→shard map
    name_to_shard = {}
    for shard_file, names in shard_names.items():
        for name in names:
            name_to_shard[name] = shard_file

    n = len(target_recs)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for i, r in enumerate(target_recs):
        ax = axes_flat[i]
        t = load_tensor(model_id, name_to_shard[r["name"]], r["name"])
        if t is None:
            ax.axis("off")
            continue

        flat = t.flatten().numpy()
        del t

        bins = np.linspace(np.min(flat), np.max(flat), 200)
        ax.hist(flat, bins=bins, density=True, color="#42a5f5", alpha=0.8, edgecolor="none")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-5)

        std = np.std(flat)
        for s_val in [3, 5]:
            ax.axvline(s_val * std, color="#ef5350", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvline(-s_val * std, color="#ef5350", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_title(f"Layer {r['layer_idx']}\nKurt={r['excess_kurtosis']:.1f}, |Max|={r['abs_max']:.2f}",
                     fontsize=8)
        ax.tick_params(labelsize=6)
        del flat
        gc.collect()

    for j in range(len(target_recs), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"{MODEL_NAME} — MLP gate_proj: Weight Distribution by Layer Depth", fontsize=13, y=1.01)
    fig.tight_layout()

    path = os.path.join(outdir, "fig6_mlp_gate_proj_evolution.png")
    save(fig, path, dpi=150)
    return path


# ─── Main ─────────────────────────────────────────

def main():
    outdir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(outdir, exist_ok=True)

    print(f"Loading shard info for {MODEL_ID}...")
    shard_names = get_shard_info(MODEL_ID)
    total_tensors = sum(len(v) for v in shard_names.values())
    print(f"Found {total_tensors} tensors across {len(shard_names)} shards")

    print("\nPhase 1: Computing statistics (tensor-by-tensor)...")
    records = compute_all_stats(MODEL_ID, shard_names)
    print(f"Computed stats for {len(records)} tensors")

    print("\n=== Generating charts ===")

    print("Fig 1: Global Distribution...")
    chart_global_distribution(MODEL_ID, shard_names, outdir)
    gc.collect()

    print("Fig 2: Per-Layer Stats...")
    chart_per_layer_stats(records, outdir)
    gc.collect()

    print("Fig 3: Worst Layer Heatmaps...")
    chart_worst_heatmaps(MODEL_ID, shard_names, records, outdir)
    gc.collect()

    print("Fig 4: Systematic Outlier Dims...")
    chart_outlier_dims(MODEL_ID, shard_names, records, outdir)
    gc.collect()

    print("Fig 5: Quantization Impact...")
    chart_quant_impact(MODEL_ID, shard_names, records, outdir)
    gc.collect()

    print("Fig 6: MLP gate_proj Evolution...")
    chart_component_evolution(MODEL_ID, shard_names, records, outdir)
    gc.collect()

    print("\n=== Done! ===")

    # Save records
    records_path = os.path.join(os.path.dirname(__file__), "analysis_data.json")
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
