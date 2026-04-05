# Repo Overview

This repo analyzes model weight outliers with model-specific pipelines.

The emphasis is on per-model analysis that can quantify outlier severity, identify systematic patterns, measure quantization impact, and produce a final written report with supporting images.

For the architectural direction of the repo, see [PROJECT_INTENT.md](PROJECT_INTENT.md).

## What It Measures

For each 2D weight tensor in a model:

- Distribution statistics: mean, std, kurtosis, dynamic range.
- Sigma-based outlier counts: 3sigma, 5sigma, 8sigma, 10sigma thresholds.
- Channel and dimension analysis: per-output-channel and per-input-dimension outlier detection.
- Quantization simulation: INT4 symmetric with and without percentile clipping.

## Current Structure

```text
misfits/
├── main.py
├── shared/
│   ├── stats.py
│   ├── io.py
│   ├── viz.py
│   └── report.py
└── models/
    ├── comparison.md
    ├── gpt2/
    ├── llama-3.2-1b/
    └── qwen3.5-0.8b/
```

Each model currently has its own `analyze.py` with model-specific classification, visualization, and reporting logic.

## Usage

```bash
# Install dependencies
pip install torch numpy matplotlib tqdm huggingface_hub safetensors

# Run all registered models
python main.py

# Run a specific model
python main.py gpt2
python main.py llama-3.2-1b

# Reuse existing stats, regenerate charts + reports
python main.py --skip-stats

# Only regenerate the comparison report
python main.py --comparison-only

# Run a model directly
python models/gpt2/analyze.py
python models/llama-3.2-1b/analyze.py --skip-stats
```

## Adding a Model

1. Create a model folder and its `analyze.py`.
2. Implement `main(skip_stats, token)` following an existing model as a template.
3. Register the model in `main.py`.
4. Run `python main.py <model-name>`.

## References

- Dettmers, "LLM.int8() and Emergent Features" (2022)
- An et al., "Systematic Outliers in Large Language Models" (ICLR 2025)
- Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs" (NeurIPS 2024)
- Liu et al., "SpinQuant: LLM quantization with learned rotations" (ICLR 2025)
- Xiao et al., "SmoothQuant" (2023)
- Lin et al., "AWQ: Activation-Aware Weight Quantization" (2024)
