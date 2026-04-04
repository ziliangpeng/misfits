"""Model loading, FP8 dequantization, and data I/O."""

import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Generator, Optional

import torch
from huggingface_hub import hf_hub_download, HfApi
from safetensors import safe_open


# ────────────────────────────────────────
# FP8 dequantization
# ────────────────────────────────────────

def dequantize_fp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 weights using block-wise scaling.

    Supports three scale layouts:
    - 1-D scale (per output channel): broadcasts over input dimension.
    - 2-D scale (block-wise): each block of input features shares one scale.
    - Scalar / other: multiplied element-wise after float conversion.

    Args:
        weight:    Raw FP8 weight tensor (any dtype, will be cast to float32).
        scale_inv: Scale-inverse tensor from the ``weight_scale_inv`` shard entry.

    Returns:
        Dequantized weight as a float32 tensor with the same shape as ``weight``.
    """
    w = weight.float()
    if scale_inv.dim() == 1:
        return w * scale_inv.float().unsqueeze(-1)
    if scale_inv.dim() == 2:
        out_f, in_f = w.shape
        num_blocks = scale_inv.shape[-1]
        block_size = in_f // num_blocks
        s = scale_inv.float().repeat_interleave(block_size, dim=-1)
        if s.shape[-1] < in_f:
            # Pad last block to cover any remainder columns
            s = torch.cat(
                [s, scale_inv[:, -1:].float().expand(-1, in_f - s.shape[-1])],
                dim=-1,
            )
        return w * s[:, :in_f]
    return w * scale_inv.float()


# ────────────────────────────────────────
# HuggingFace model index
# ────────────────────────────────────────

def get_weight_map(model_id: str, token: Optional[str] = None) -> dict[str, str]:
    """Get a tensor-name → shard-filename mapping for a HuggingFace model.

    Supports both single-file (``model.safetensors``) and sharded
    (``model.safetensors.index.json``) models.

    Args:
        model_id: HuggingFace model ID, e.g. ``"deepseek-ai/DeepSeek-V3"``.
        token:    Optional HuggingFace access token for gated models.

    Returns:
        A dict mapping each tensor name to the shard filename that contains it.

    Raises:
        FileNotFoundError: If no safetensors files are found in the repository.
    """
    api = HfApi()
    files = [
        f.rfilename
        for f in api.list_repo_tree(model_id, token=token)
        if hasattr(f, "rfilename")
    ]

    if "model.safetensors.index.json" in files:
        idx_path = hf_hub_download(model_id, "model.safetensors.index.json", token=token)
        with open(idx_path) as fh:
            weight_map: dict[str, str] = json.load(fh)["weight_map"]
    elif "model.safetensors" in files:
        p = hf_hub_download(model_id, "model.safetensors", token=token)
        with safe_open(p, framework="pt", device="cpu") as sf:
            weight_map = {k: "model.safetensors" for k in sf.keys()}
    else:
        raise FileNotFoundError(f"No safetensors files found in {model_id!r}")

    return weight_map


# ────────────────────────────────────────
# Streaming tensor iteration
# ────────────────────────────────────────

def iter_tensors(
    model_id: str,
    weight_map: dict[str, str],
    names: list[str],
    token: Optional[str] = None,
    scale_map: Optional[dict[str, str]] = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield ``(name, tensor)`` pairs, streaming one shard at a time.

    Tensors are grouped by shard to minimise downloads. Each shard is opened
    once, all requested tensors are read, then the shard handle is released and
    ``gc.collect()`` is called to free memory before moving to the next shard.

    FP8 dequantization is applied automatically when ``scale_map`` is provided
    and the tensor has a matching entry.

    Args:
        model_id:   HuggingFace model ID.
        weight_map: Mapping from tensor name to shard filename (from
                    :func:`get_weight_map`).
        names:      Ordered list of tensor names to yield.
        token:      Optional HuggingFace access token.
        scale_map:  Optional mapping from *weight* tensor name to the
                    corresponding ``weight_scale_inv`` tensor name.
                    When provided, FP8 tensors are dequantized before yielding.

    Yields:
        ``(name, tensor)`` tuples in the order determined by shard grouping.
        The caller should ``del tensor`` after use to keep peak memory low.
    """
    scale_map = scale_map or {}

    # Group requested names by shard file
    shard_targets: dict[str, list[str]] = defaultdict(list)
    for n in names:
        shard_targets[weight_map[n]].append(n)

    for shard_file in sorted(shard_targets):
        tensor_names = shard_targets[shard_file]
        shard_path = hf_hub_download(model_id, shard_file, token=token)

        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for tname in tensor_names:
                w = sf.get_tensor(tname)

                # FP8 dequantization
                if tname in scale_map:
                    scale_tensor_name = scale_map[tname]
                    scale_shard = weight_map.get(scale_tensor_name, shard_file)
                    if scale_shard == shard_file:
                        si_tensor = sf.get_tensor(scale_tensor_name)
                    else:
                        sp = hf_hub_download(model_id, scale_shard, token=token)
                        with safe_open(sp, framework="pt", device="cpu") as sf2:
                            si_tensor = sf2.get_tensor(scale_tensor_name)
                    w = dequantize_fp8(w, si_tensor)
                    del si_tensor

                yield tname, w
                del w

        gc.collect()


# ────────────────────────────────────────
# Single-tensor loader
# ────────────────────────────────────────

def load_tensor(
    model_id: str,
    shard_file: str,
    name: str,
    token: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Load a single tensor by name from a specific shard.

    Args:
        model_id:   HuggingFace model ID.
        shard_file: Shard filename within the repository (e.g.
                    ``"model-00001-of-00163.safetensors"``).
        name:       Tensor name to load.
        token:      Optional HuggingFace access token.

    Returns:
        The requested tensor, or ``None`` if the name is not present in the shard.
    """
    shard_path = hf_hub_download(model_id, shard_file, token=token)
    with safe_open(shard_path, framework="pt", device="cpu") as sf:
        if name in sf.keys():
            return sf.get_tensor(name)
    return None


# ────────────────────────────────────────
# JSONL helpers
# ────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, skipping error-only records and malformed lines.

    A record is treated as error-only (and skipped) if it contains an
    ``"error"`` key but no ``"std"`` key — i.e. it was written as a
    failure placeholder by the analysis loop.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of successfully parsed, non-error records.
    """
    records: list[dict] = []
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                if "error" in rec and "std" not in rec:
                    continue  # skip failed-tensor placeholders
                records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def save_jsonl(records: list[dict], path: str) -> None:
    """Save a list of records as JSONL (one JSON object per line).

    Creates parent directories as needed. Overwrites any existing file.

    Args:
        records: List of JSON-serialisable dicts.
        path:    Destination file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def load_done_set(path: str) -> set[str]:
    """Load the set of already-processed tensor names from a JSONL file.

    Used to implement resume support: tensors whose names are in the returned
    set have already been analyzed and written to the output file.

    Args:
        path: Path to an existing output ``.jsonl`` file.
              Returns an empty set if the file does not exist.

    Returns:
        Set of ``tensor_name`` strings found in the file.
    """
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["tensor_name"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done
