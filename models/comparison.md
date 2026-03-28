# Weight Outlier Comparison Across Models

A cross-model comparison of weight distribution outlier patterns.

## Models Analyzed

| | GPT-2 | Qwen3.5-0.8B | Llama 3.2 1B |
|---|---|---|---|
| **Parameters** | 124M | 818M (619M analyzed) | 1.2B (973M analyzed) |
| **Architecture** | Dense Transformer | Hybrid: Gated DeltaNet + Attention | Dense Transformer (GQA) |
| **Layers** | 12 | 24 | 16 |
| **Hidden dim** | 768 | 1024 | 2048 |
| **Tensors analyzed** | 48 | 264 | 112 |
| **Release** | 2019 | 2026 | 2024 |

## Key Metrics

| Metric | GPT-2 | Qwen3.5-0.8B | Llama 3.2 1B |
|--------|------:|-------------:|-------------:|
| **Global \|max weight\|** | 17.10 | 4.75 | 1.23 |
| **Median \|max weight\|** | 2.43 | 0.21 | 0.43 |
| **Peak kurtosis** | 790.5 | 712.0 | 16.1 |
| **Median kurtosis** | 3.24 | 1.85 | 1.61 |
| **Mean kurtosis** | 61.4 | 5.9 | 2.2 |
| **P99 kurtosis** | 774.6 | 32.6 | 8.4 |
| **Median dynamic range** | 33.0 | 26.2 | 33.4 |
| **Max dynamic range** | 305.1 | 368.2 | 98.0 |
| **Mean % beyond 3σ** | 0.643% | 0.988% | 0.739% |
| **Mean % beyond 10σ** | 0.0086% | 0.0025% | 0.0016% |
| **Mean INT4 clip improvement** | 90.8% | 83.0% | 90.4% |
| **Max INT4 clip improvement** | 96.9% | 97.8% | 97.8% |

## Component Comparison

### GPT-2

| Component | Count | Mean Kurtosis | Max Kurtosis | Max \|weight\| |
|-----------|------:|--------------:|-------------:|--------------:|
| attention | 24 | 18.3 | 190.1 | 8.88 |
| mlp_dense | 24 | 104.5 | **790.5** | **17.10** |

GPT-2's outlier problem is dominated by **MLP layers**, especially `c_proj` in early layers (layers 0–3). These have kurtosis 400–800, driven by a few extreme output channels.

### Qwen3.5-0.8B

| Component | Count | Mean Kurtosis | Max Kurtosis | Max \|weight\| |
|-----------|------:|--------------:|-------------:|--------------:|
| attention (standard) | 28 | 3.2 | 8.8 | 0.60 |
| mlp_dense | 99 | 1.7 | 10.0 | 0.51 |
| linear_attn (DeltaNet) | 108 | 4.6 | 48.5 | 1.82 |
| visual (encoder) | 52 | 2.9 | 37.1 | **4.75** |
| mtp (multi-token pred) | 8 | 93.3 | **712.0** | 0.60 |

Standard attention and MLP are extremely well-behaved. The **Gated DeltaNet layers** have moderate kurtosis. A single `mtp.fc` tensor has kurtosis 712 — an isolated extreme outlier.

### Llama 3.2 1B

| Component | Count | Mean Kurtosis | Max Kurtosis | Max \|weight\| |
|-----------|------:|--------------:|-------------:|--------------:|
| attention | 64 | 2.6 | 8.4 | 1.13 |
| mlp_dense | 48 | 1.8 | **16.1** | **1.23** |

Llama 3.2 is **remarkably clean** — no tensor exceeds kurtosis 16.1, and the global max weight is only 1.23. This is the most quantization-friendly model of the three by a wide margin.

## Systematic Outlier Dimensions

| | GPT-2 | Qwen3.5-0.8B | Llama 3.2 1B |
|---|---|---|---|
| **Top outlier dim** | dim 447 (50%) | dim 74 (23%) | dim 1107 (15%) |
| **Runner-ups** | 481 (30%), 373 (28%) | 8 (10%), 42 (9%) | 352 (13%), 1511 (10%) |
| **Pattern** | Sharply concentrated | Moderately diffuse | Diffuse |

Newer models show progressively less concentrated outlier dimensions — likely a sign of improved training practices.

## Evolution of Quantization Friendliness

Clear generational improvement:

1. **GPT-2 (2019)**: Severe outliers. MLP c_proj layers are quantization disasters (kurtosis ~800). Requires outlier-aware quantization (GPTQ, SmoothQuant, etc.)

2. **Qwen3.5-0.8B (2026)**: Mixed. Standard transformer components are near-Gaussian. But novel architecture components (DeltaNet, MTP) introduce new outlier patterns that need fresh quantization strategies.

3. **Llama 3.2 1B (2024)**: Excellent. Near-Gaussian everywhere, max kurtosis only 16. Almost any quantization method should work well. Likely benefits from Meta's quantization-aware training and architectural improvements.

## Detailed Reports

- [GPT-2](gpt2/gpt2_weight_outlier_analysis.md)
- [Qwen3.5-0.8B](qwen3.5-0.8b/qwen3_5_0_8b_weight_outlier_analysis.md)
- [Llama 3.2 1B](llama-3.2-1b/llama_3_2_1b_weight_outlier_analysis.md)
