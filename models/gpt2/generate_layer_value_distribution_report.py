#!/usr/bin/env python3
"""Generate a simple GPT-2 report of per-layer value distributions.

This is an intentionally simple first-pass report:

- download/load the smallest GPT-2 checkpoint
- iterate through every stored tensor in the model
- generate one value-distribution histogram per tensor
- write a Markdown report that embeds all generated plots
- write a JSON data file with basic per-tensor statistics

Usage:
    python -m models.gpt2.generate_layer_value_distribution_report
    python -m models.gpt2.generate_layer_value_distribution_report --limit 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.gpt2 import GPT2Weights


MODEL_DIR = Path(__file__).resolve().parent
IMAGE_DIR = MODEL_DIR / "layer_value_distribution_images"
REPORT_PATH = MODEL_DIR / "gpt2_layer_value_distribution_report.md"
DATA_PATH = MODEL_DIR / "gpt2_layer_value_distribution_data.json"


def _slugify_layer_name(layer_name: str) -> str:
    """Convert a tensor name into a stable filename stem."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", layer_name).replace(".", "_")


def _tensor_stats(tensor: torch.Tensor) -> dict:
    """Compute a small set of simple tensor statistics."""
    values = tensor.detach().cpu().float().reshape(-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def _save_distribution_plot(layer_name: str, tensor: torch.Tensor, image_dir: Path) -> str:
    """Save a histogram of tensor values and return the image filename."""
    values = tensor.detach().cpu().float().reshape(-1).numpy()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(values, bins=120, density=True, color="#1f77b4", alpha=0.9)
    ax.set_title(layer_name)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)

    image_name = f"{_slugify_layer_name(layer_name)}.png"
    image_path = image_dir / image_name
    fig.tight_layout()
    fig.savefig(image_path, dpi=120)
    plt.close(fig)

    return image_name


def generate_report(
    model: GPT2Weights,
    image_dir: Path,
    report_path: Path,
    data_path: Path,
    limit: int | None = None,
) -> list[dict]:
    """Generate per-layer plots, a JSON data file, and a Markdown report."""
    image_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    tensors = model.iter_tensors()
    total = min(limit, len(model)) if limit is not None else len(model)

    for idx, (layer_name, tensor) in enumerate(
        tqdm(tensors, total=total, desc="Plotting layers"),
        start=1,
    ):
        if limit is not None and idx > limit:
            break

        stats = _tensor_stats(tensor)
        image_name = _save_distribution_plot(layer_name, tensor, image_dir)
        records.append({
            "layer_name": layer_name,
            "image": image_name,
            **stats,
        })

    data_path.write_text(json.dumps(records, indent=2))

    lines = [
        "# GPT-2 Layer Value Distribution Report",
        "",
        "A simple first-pass report showing the raw value distribution of each stored GPT-2 tensor.",
        "",
        f"- Model: `openai-community/gpt2`",
        f"- Tensors included: `{len(records)}`",
        f"- Images directory: `{image_dir.name}`",
        "",
        "## Layers",
        "",
    ]

    for record in records:
        lines.extend([
            f"### `{record['layer_name']}`",
            "",
            f"- Shape: `{tuple(record['shape'])}`",
            f"- Dtype: `{record['dtype']}`",
            f"- Numel: `{record['numel']}`",
            f"- Min: `{record['min']:.6f}`",
            f"- Max: `{record['max']:.6f}`",
            f"- Mean: `{record['mean']:.6f}`",
            f"- Std: `{record['std']:.6f}`",
            "",
            f"![{record['layer_name']}]({image_dir.name}/{record['image']})",
            "",
        ])

    report_path.write_text("\n".join(lines))
    return records


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a simple GPT-2 layer value distribution report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N tensors. Useful for smoke tests.",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional local download directory passed to Hugging Face.",
    )
    args = parser.parse_args()

    model = GPT2Weights.from_huggingface(local_dir=args.local_dir)
    records = generate_report(
        model=model,
        image_dir=IMAGE_DIR,
        report_path=REPORT_PATH,
        data_path=DATA_PATH,
        limit=args.limit,
    )

    print(f"Generated {len(records)} layer plots")
    print(f"Data: {DATA_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
