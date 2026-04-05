"""Fresh GPT-2 weight access module.

This module is intentionally separate from the existing GPT-2 analysis code.
It provides a minimal, incremental foundation for:

- downloading the smallest GPT-2 model from Hugging Face
- loading its weights into memory
- listing available weight names
- retrieving a tensor by layer name
"""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterator
from typing import Iterable

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


GPT2_MODEL_ID = "openai-community/gpt2"


def download_gpt2_snapshot(
    local_dir: str | Path | None = None,
    token: str | None = None,
) -> Path:
    """Download the smallest GPT-2 model snapshot from Hugging Face.

    The download is limited to the config file and weight files needed to load
    the model parameters. Hugging Face returns the local snapshot directory.

    Args:
        local_dir: Optional explicit destination directory. If omitted,
            Hugging Face uses its cache location and returns that snapshot path.
        token: Optional Hugging Face access token.

    Returns:
        Local path to the downloaded snapshot directory.
    """
    snapshot_path = snapshot_download(
        repo_id=GPT2_MODEL_ID,
        local_dir=str(local_dir) if local_dir is not None else None,
        token=token,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "model-*.safetensors",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "pytorch_model-*.bin",
        ],
    )
    return Path(snapshot_path)


def _load_safetensors_files(weight_files: Iterable[Path]) -> dict[str, torch.Tensor]:
    """Load one or more safetensors files into a single state dict."""
    state_dict: dict[str, torch.Tensor] = {}
    for weight_file in weight_files:
        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                state_dict[key] = handle.get_tensor(key)
    return state_dict


def _load_bin_files(weight_files: Iterable[Path]) -> dict[str, torch.Tensor]:
    """Load one or more PyTorch checkpoint shards into a single state dict."""
    state_dict: dict[str, torch.Tensor] = {}
    for weight_file in weight_files:
        shard = torch.load(weight_file, map_location="cpu")
        if not isinstance(shard, dict):
            raise TypeError(f"Unexpected checkpoint format in {weight_file}")
        state_dict.update(shard)
    return state_dict


def _resolve_safetensors_files(snapshot_dir: Path) -> list[Path]:
    """Resolve the safetensors files present in a snapshot directory."""
    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as fh:
            weight_map = json.load(fh)["weight_map"]
        return sorted({snapshot_dir / shard_name for shard_name in weight_map.values()})

    weight_file = snapshot_dir / "model.safetensors"
    if weight_file.exists():
        return [weight_file]

    return []


def _resolve_bin_files(snapshot_dir: Path) -> list[Path]:
    """Resolve the PyTorch checkpoint files present in a snapshot directory."""
    index_path = snapshot_dir / "pytorch_model.bin.index.json"
    if index_path.exists():
        with index_path.open() as fh:
            weight_map = json.load(fh)["weight_map"]
        return sorted({snapshot_dir / shard_name for shard_name in weight_map.values()})

    weight_file = snapshot_dir / "pytorch_model.bin"
    if weight_file.exists():
        return [weight_file]

    return []


def load_gpt2_state_dict(snapshot_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load GPT-2 weights from a downloaded snapshot into memory.

    Safetensors is preferred when present. PyTorch `.bin` weights are used as a
    fallback for repositories that do not ship safetensors files.

    Args:
        snapshot_dir: Path returned by `download_gpt2_snapshot()`.

    Returns:
        A state dict mapping tensor names to CPU tensors.
    """
    snapshot_path = Path(snapshot_dir)

    safetensors_files = _resolve_safetensors_files(snapshot_path)
    if safetensors_files:
        return _load_safetensors_files(safetensors_files)

    bin_files = _resolve_bin_files(snapshot_path)
    if bin_files:
        return _load_bin_files(bin_files)

    raise FileNotFoundError(
        f"No GPT-2 weight files found in snapshot directory: {snapshot_path}"
    )


class GPT2Weights:
    """In-memory GPT-2 weight access.

    This first version loads the entire GPT-2 state dict into CPU memory.
    Later iterations can add lazy or per-layer loading without changing the
    external interface used by analysis code.
    """

    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        snapshot_dir: str | Path | None = None,
    ) -> None:
        self._state_dict = state_dict
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else None

    @classmethod
    def from_snapshot(cls, snapshot_dir: str | Path) -> "GPT2Weights":
        """Build a GPT-2 weight wrapper from an existing local snapshot."""
        snapshot_path = Path(snapshot_dir)
        state_dict = load_gpt2_state_dict(snapshot_path)
        return cls(state_dict=state_dict, snapshot_dir=snapshot_path)

    @classmethod
    def from_huggingface(
        cls,
        local_dir: str | Path | None = None,
        token: str | None = None,
    ) -> "GPT2Weights":
        """Download GPT-2 from Hugging Face and load the weights into memory."""
        snapshot_dir = download_gpt2_snapshot(local_dir=local_dir, token=token)
        return cls.from_snapshot(snapshot_dir)

    def list_layers(self) -> list[str]:
        """Return all raw checkpoint tensor names in sorted order."""
        return sorted(self._state_dict.keys())

    def _resolve_layer_name(self, layer_name: str) -> str:
        """Resolve a requested layer name to a raw checkpoint tensor name.

        GPT-2 checkpoints from `openai-community/gpt2` use names like
        `wte.weight` and `h.0.attn.c_attn.weight`. This resolver also accepts
        the common `transformer.*` prefix and maps it back to the stored key.
        """
        if layer_name in self._state_dict:
            return layer_name

        if layer_name.startswith("transformer."):
            candidate = layer_name.removeprefix("transformer.")
            if candidate in self._state_dict:
                return candidate

        raise KeyError(f"Unknown GPT-2 layer: {layer_name}")

    def get_tensor(self, layer_name: str) -> torch.Tensor:
        """Return the tensor for a given layer name.

        Accepts either the raw checkpoint name or the equivalent
        `transformer.*` name when applicable.
        """
        resolved_name = self._resolve_layer_name(layer_name)
        return self._state_dict[resolved_name]

    def iter_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield `(name, tensor)` pairs for all stored tensors in sorted order."""
        for layer_name in self.list_layers():
            yield layer_name, self._state_dict[layer_name]

    def __contains__(self, layer_name: str) -> bool:
        """Return whether a layer name can be resolved in the loaded state dict."""
        try:
            self._resolve_layer_name(layer_name)
        except KeyError:
            return False
        return True

    def __len__(self) -> int:
        """Return the number of tensors currently loaded."""
        return len(self._state_dict)
