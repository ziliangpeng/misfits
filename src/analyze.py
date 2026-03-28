#!/usr/bin/env python3
"""
Step 1: Collect per-tensor statistics from a HuggingFace model.

Streams one tensor at a time from safetensors shards.
Writes results incrementally to JSONL — fully resumable on restart.
Handles FP8 models (DeepSeek V3) via automatic dequantization.
"""

import argparse
import json
import os
import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download, HfApi
from safetensors import safe_open


# ────────────────────────────────────────
# Filtering
# ────────────────────────────────────────

def should_analyze(name: str) -> bool:
    """Keep only 2D weight matrices relevant to quantization."""
    if "scale_inv" in name:
        return False
    if "layernorm" in name or "rmsnorm" in name or "norm.weight" in name:
        return False
    if "embed_tokens" in name or "embed_positions" in name or "wte." in name or "wpe." in name:
        return False
    # MoE router gate (small, routing logic)
    if name.endswith(".gate.weight") and "mlp.gate.weight" in name:
        return False
    if "e_score_correction" in name:
        return False
    if not name.endswith(".weight"):
        return False
    return True


def classify(name: str) -> dict:
    """Extract structured metadata from tensor name."""
    parts = name.split(".")
    info = {"tensor_name": name, "layer_idx": None, "component": "other",
            "proj_type": parts[-2] if len(parts) >= 2 else "unknown", "expert_idx": None}

    for i, p in enumerate(parts):
        if p in ("layers", "h") and i + 1 < len(parts):
            try:
                info["layer_idx"] = int(parts[i + 1])
            except ValueError:
                pass
            break

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
# FP8 dequantization
# ────────────────────────────────────────

def dequantize_fp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    w = weight.float()
    if scale_inv.dim() == 1:
        return w * scale_inv.float().unsqueeze(-1)
    if scale_inv.dim() == 2:
        out_f, in_f = w.shape
        num_blocks = scale_inv.shape[-1]
        block_size = in_f // num_blocks
        s = scale_inv.float().repeat_interleave(block_size, dim=-1)
        if s.shape[-1] < in_f:
            s = torch.cat([s, scale_inv[:, -1:].float().expand(-1, in_f - s.shape[-1])], dim=-1)
        return w * s[:, :in_f]
    return w * scale_inv.float()


# ────────────────────────────────────────
# Per-tensor statistics
# ────────────────────────────────────────

def compute_stats(t: torch.Tensor) -> dict:
    """Comprehensive outlier statistics for one weight tensor."""
    t = t.float().cpu()
    flat = t.flatten()
    n = flat.numel()

    mean = flat.mean().item()
    std = flat.std().item()
    abs_max = flat.abs().max().item()
    median_abs = flat.abs().median().item()

    # Sigma-based outlier counts
    dev = (flat - mean).abs()
    outliers = {}
    for s in [3, 5, 8, 10]:
        c = (dev > s * std).sum().item()
        outliers[f"beyond_{s}sigma_count"] = int(c)
        outliers[f"beyond_{s}sigma_pct"] = c / n * 100

    # Kurtosis
    kurtosis = ((flat - mean) / std).pow(4).mean().item() - 3.0 if std > 0 else 0.0

    # Channel-wise (per output row)
    ch = {}
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
    idim = {}
    if t.dim() == 2 and t.shape[1] > 1:
        col_max = t.abs().amax(dim=0)
        col_med = col_max.median().item()
        outlier_cols = (col_max > 3 * col_med).nonzero(as_tuple=True)[0].tolist()
        idim = {
            "input_dim_median_abs_max": col_med,
            "input_dim_max_abs_max": col_max.max().item(),
            "input_dim_outlier_count": len(outlier_cols),
            "input_dim_outlier_indices": outlier_cols[:30],
        }

    # INT4 quantization error
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

    return {
        "shape": list(t.shape), "num_params": n,
        "mean": mean, "std": std, "min": flat.min().item(), "max": flat.max().item(),
        "abs_max": abs_max, "median_abs": median_abs,
        "dynamic_range": abs_max / median_abs if median_abs > 0 else float("inf"),
        "excess_kurtosis": kurtosis,
        **outliers, **ch, **idim, **qerr,
    }


# ────────────────────────────────────────
# Main loop
# ────────────────────────────────────────

def run(model_id: str, output_path: str, token: str = None):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Resume support
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["tensor_name"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resume: {len(done)} tensors already done")

    # Index
    api = HfApi()
    files = [f.rfilename for f in api.list_repo_tree(model_id, token=token)
             if hasattr(f, 'rfilename')]

    if "model.safetensors.index.json" in files:
        idx_path = hf_hub_download(model_id, "model.safetensors.index.json", token=token)
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
    elif "model.safetensors" in files:
        p = hf_hub_download(model_id, "model.safetensors", token=token)
        with safe_open(p, framework="pt", device="cpu") as sf:
            weight_map = {k: "model.safetensors" for k in sf.keys()}
    else:
        raise FileNotFoundError(f"No safetensors in {model_id}")

    # Scale map for FP8
    scale_map = {}
    for name in weight_map:
        if "weight_scale_inv" in name:
            scale_map[name.replace("_scale_inv", "")] = name

    # Filter & group by shard
    targets = [n for n in weight_map if should_analyze(n) and n not in done]
    shard_targets = defaultdict(list)
    for n in targets:
        shard_targets[weight_map[n]].append(n)

    print(f"Tensors remaining: {len(targets)} across {len(shard_targets)} shards")

    processed = 0
    for si, shard_file in enumerate(sorted(shard_targets)):
        names = shard_targets[shard_file]
        print(f"\n[{si+1}/{len(shard_targets)}] {shard_file} ({len(names)} tensors)")

        shard_path = hf_hub_download(model_id, shard_file, token=token)

        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for tname in tqdm(names, desc="  Analyzing"):
                try:
                    w = sf.get_tensor(tname)

                    is_fp8 = tname in scale_map
                    if is_fp8:
                        sn = scale_map[tname]
                        ss = weight_map[sn]
                        if ss == shard_file:
                            si_tensor = sf.get_tensor(sn)
                        else:
                            sp = hf_hub_download(model_id, ss, token=token)
                            with safe_open(sp, framework="pt", device="cpu") as sf2:
                                si_tensor = sf2.get_tensor(sn)
                        w = dequantize_fp8(w, si_tensor)
                        del si_tensor

                    if w.dim() < 2:
                        del w
                        continue

                    meta = classify(tname)
                    stats = compute_stats(w)
                    record = {**meta, **stats, "is_fp8_dequantized": is_fp8}

                    with open(output_path, "a") as f:
                        f.write(json.dumps(record) + "\n")

                    processed += 1
                    del w
                except Exception as e:
                    print(f"  ERROR {tname}: {e}")
                    with open(output_path, "a") as f:
                        f.write(json.dumps({"tensor_name": tname, "error": str(e)}) + "\n")

                gc.collect()

        gc.collect()
        print(f"  Total done: {len(done) + processed}/{len(done) + len(targets)}")

    print(f"\nComplete. {processed} new tensors analyzed. Output: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--token", default=None)
    args = p.parse_args()
    run(args.model, args.output, args.token)
