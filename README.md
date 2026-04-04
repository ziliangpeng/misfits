# misfits

Empirical weight outlier analysis for large language models — finding the misfits that break quantization.

LLM weights contain extreme outliers that cause massive quantization error under INT4/FP4 formats. This project provides per-model analysis tools that quantify outlier severity, identify systematic patterns, and measure quantization impact.

## Models

| Model | Architecture | Params | Key Finding | Report |
|-------|-------------|--------|-------------|--------|
| GPT-2 | Dense Transformer | 124M | Severe MLP c_proj outliers (kurtosis ~800) | [→ Report](models/gpt2/gpt2_weight_outlier_analysis.md) |
| Llama 3.2 1B | Dense Transformer (GQA) | 1.2B | Remarkably clean — max kurtosis 16 | [→ Report](models/llama-3.2-1b/llama_3_2_1b_weight_outlier_analysis.md) |
| Qwen3.5-0.8B | Hybrid DeltaNet + Attention | 818M | Novel arch shows new outlier patterns | [→ Report](models/qwen3.5-0.8b/qwen3_5_0_8b_weight_outlier_analysis.md) |

[Full comparison →](models/comparison.md)

## What It Measures

For each 2D weight tensor in a model:
- **Distribution statistics**: mean, std, kurtosis, dynamic range
- **Sigma-based outlier counts**: 3σ, 5σ, 8σ, 10σ thresholds
- **Channel and dimension analysis**: per-output-channel and per-input-dimension outlier detection
- **Quantization simulation**: INT4 symmetric with and without percentile clipping

## Project Structure

```
misfits/
├── main.py              # Master entry point — runs all models
├── shared/              # Reusable analysis utilities
│   ├── stats.py         #   Per-tensor statistical computations
│   ├── io.py            #   Model loading, FP8 dequant, data I/O
│   ├── viz.py           #   Chart style and helpers
│   └── report.py        #   Report generation
└── models/
    ├── comparison.md    # Cross-model comparison
    ├── gpt2/
    │   ├── analyze.py   # GPT-2 specific analysis + charts
    │   ├── images/      # Generated visualizations
    │   └── *.md         # Analysis report
    ├── llama-3.2-1b/
    │   └── ...
    └── qwen3.5-0.8b/
        └── ...
```

Each model has its own `analyze.py` with model-specific classification, visualization, and narrative.

## Usage

```bash
# Install dependencies
pip install torch numpy matplotlib tqdm huggingface_hub safetensors

# Run all models
python main.py

# Run a specific model
python main.py gpt2
python main.py llama-3.2-1b

# Skip stats collection (reuse existing data), regenerate charts + reports
python main.py --skip-stats

# Only regenerate the comparison report
python main.py --comparison-only

# Run individual model directly
python models/gpt2/analyze.py
python models/llama-3.2-1b/analyze.py --skip-stats
```

## Adding a New Model

1. Create `models/<model-name>/analyze.py`
2. Implement `main(skip_stats, token)` following the existing models as template
3. Register it in `main.py`'s `MODELS` dict
4. Run `python main.py <model-name>`

## References

- Dettmers, "LLM.int8() and Emergent Features" (2022)
- An et al., "Systematic Outliers in Large Language Models" (ICLR 2025)
- Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs" (NeurIPS 2024)
- Liu et al., "SpinQuant: LLM quantization with learned rotations" (ICLR 2025)
- Xiao et al., "SmoothQuant" (2023)
- Lin et al., "AWQ: Activation-Aware Weight Quantization" (2024)
