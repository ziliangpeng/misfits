"""Per-tensor statistical analysis for weight outlier detection."""

import torch
import numpy as np
from typing import Optional


# ────────────────────────────────────────
# Tensor filtering
# ────────────────────────────────────────

def should_analyze(name: str) -> bool:
    """Return True if this tensor should be included in outlier analysis.

    Keeps only 2D weight matrices relevant to quantization.
    Skips layer norms, embeddings, biases, routing gates, and FP8 scale tensors.

    Args:
        name: Fully-qualified tensor name (e.g. ``model.layers.0.self_attn.q_proj.weight``).

    Returns:
        ``True`` if the tensor should be analyzed, ``False`` otherwise.
    """
    if "scale_inv" in name:
        return False
    if "layernorm" in name or "rmsnorm" in name or "norm.weight" in name:
        return False
    if "embed_tokens" in name or "embed_positions" in name or "wte." in name or "wpe." in name:
        return False
    # MoE router gate (small, routing logic — not a linear projection)
    if name.endswith(".gate.weight") and "mlp.gate.weight" in name:
        return False
    if "e_score_correction" in name:
        return False
    if not name.endswith(".weight"):
        return False
    return True


# ────────────────────────────────────────
# Tensor classification
# ────────────────────────────────────────

def classify(name: str) -> dict:
    """Extract structured metadata from a tensor name.

    Parses the dot-separated tensor name and infers the layer index,
    architectural component, projection type, and (for MoE models)
    the expert index.

    Args:
        name: Fully-qualified tensor name.

    Returns:
        A dict with keys:
            - ``tensor_name``  (str)  — original name
            - ``layer_idx``    (int | None) — transformer layer number
            - ``component``    (str) — one of ``attention``, ``mlp_expert``,
              ``mlp_shared``, ``mlp_dense``, ``lm_head``, ``other``
            - ``proj_type``    (str) — second-to-last name segment (e.g. ``q_proj``)
            - ``expert_idx``   (int | None) — MoE expert number, if applicable
    """
    parts = name.split(".")
    info: dict = {
        "tensor_name": name,
        "layer_idx": None,
        "component": "other",
        "proj_type": parts[-2] if len(parts) >= 2 else "unknown",
        "expert_idx": None,
    }

    # Extract transformer layer index
    for i, p in enumerate(parts):
        if p in ("layers", "h") and i + 1 < len(parts):
            try:
                info["layer_idx"] = int(parts[i + 1])
            except ValueError:
                pass
            break

    # Classify architectural component
    if any(k in name for k in ("self_attn", "attention", "attn.c_attn", "attn.c_proj")):
        info["component"] = "attention"
    elif "experts." in name:
        info["component"] = "mlp_expert"
        for i, p in enumerate(parts):
            if p == "experts" and i + 1 < len(parts):
                try:
                    info["expert_idx"] = int(parts[i + 1])
                except ValueError:
                    pass
                break
    elif "shared_expert" in name:
        info["component"] = "mlp_shared"
    elif "mlp" in name:
        info["component"] = "mlp_dense"
    elif "lm_head" in name:
        info["component"] = "lm_head"

    return info


# ────────────────────────────────────────
# Per-tensor statistics
# ────────────────────────────────────────

def compute_stats(t: torch.Tensor) -> dict:
    """Comprehensive outlier statistics for one weight tensor.

    Computes distribution stats, sigma-based outlier counts,
    channel-wise and input-dimension outlier detection,
    and INT4 quantization error simulation.

    Args:
        t: Weight tensor of any shape. Will be cast to float32 on CPU.
           Non-2D tensors will have reduced channel/input-dim analysis.

    Returns:
        A dict containing:
            - Basic distribution: ``shape``, ``num_params``, ``mean``, ``std``,
              ``min``, ``max``, ``abs_max``, ``median_abs``, ``dynamic_range``,
              ``excess_kurtosis``
            - Sigma outliers (for thresholds 3, 5, 8, 10):
              ``beyond_Nsigma_count``, ``beyond_Nsigma_pct``
            - Channel-wise (if 2D, rows > 1): ``num_channels``,
              ``channel_median_abs_max``, ``channel_max_abs_max``,
              ``outlier_channel_count``, ``outlier_channel_pct``
            - Input-dim (if 2D, cols > 1): ``input_dim_median_abs_max``,
              ``input_dim_max_abs_max``, ``input_dim_outlier_count``,
              ``input_dim_outlier_indices``
            - INT4 quant error (if ``abs_max > 0``): ``int4_naive_mse``,
              ``int4_clip999_mse``, ``int4_clip_improvement_pct``
    """
    t = t.float().cpu()
    flat = t.flatten()
    n = flat.numel()

    mean = flat.mean().item()
    std = flat.std().item()
    abs_max = flat.abs().max().item()
    median_abs = flat.abs().median().item()

    # Sigma-based outlier counts
    dev = (flat - mean).abs()
    outliers: dict = {}
    for s in [3, 5, 8, 10]:
        c = (dev > s * std).sum().item()
        outliers[f"beyond_{s}sigma_count"] = int(c)
        outliers[f"beyond_{s}sigma_pct"] = c / n * 100

    # Excess kurtosis (Fisher definition: 0 for a Gaussian)
    kurtosis = ((flat - mean) / std).pow(4).mean().item() - 3.0 if std > 0 else 0.0

    # Channel-wise (per output row) outlier detection
    ch: dict = {}
    if t.dim() == 2 and t.shape[0] > 1:
        ch_max = t.abs().amax(dim=1)
        ch_med = ch_max.median().item()
        outlier_mask = ch_max > 3 * ch_med
        ch = {
            "num_channels": t.shape[0],
            "channel_median_abs_max": ch_med,
            "channel_max_abs_max": ch_max.max().item(),
            "outlier_channel_count": int(outlier_mask.sum().item()),
            "outlier_channel_pct": outlier_mask.sum().item() / t.shape[0] * 100,
        }

    # Input-dimension (per column) outlier detection
    idim: dict = {}
    if t.dim() == 2 and t.shape[1] > 1:
        col_max = t.abs().amax(dim=0)
        col_med = col_max.median().item()
        outlier_cols = (col_max > 3 * col_med).nonzero(as_tuple=True)[0].tolist()
        idim = {
            "input_dim_median_abs_max": col_med,
            "input_dim_max_abs_max": col_max.max().item(),
            "input_dim_outlier_count": len(outlier_cols),
            "input_dim_outlier_indices": outlier_cols[:30],  # cap to 30 for storage
        }

    # INT4 symmetric quantization error simulation
    qerr: dict = {}
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
            "int4_clip_improvement_pct": (
                (mse_naive - mse_clip) / mse_naive * 100 if mse_naive > 0 else 0
            ),
        }

    return {
        "shape": list(t.shape),
        "num_params": n,
        "mean": mean,
        "std": std,
        "min": flat.min().item(),
        "max": flat.max().item(),
        "abs_max": abs_max,
        "median_abs": median_abs,
        "dynamic_range": abs_max / median_abs if median_abs > 0 else float("inf"),
        "excess_kurtosis": kurtosis,
        **outliers,
        **ch,
        **idim,
        **qerr,
    }
