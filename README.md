# misfits

Automated weight outlier analysis for LLMs — finding the misfits that break quantization.

One command in, full analysis out: per-tensor statistics, visualizations, and a markdown report for any HuggingFace model.

## Quick Start

```bash
pip install torch numpy matplotlib tqdm huggingface_hub safetensors

# Analyze any model
python src/run.py openai-community/gpt2
python src/run.py Qwen/Qwen3.5-0.8B
python src/run.py meta-llama/Llama-3.2-1B --token hf_xxx
```

## What It Does

LLM weights often have extreme outliers — values 100x+ larger than the median. These outliers cause massive quantization error when using INT4/INT8 formats. This tool finds them.

For each 2D weight tensor in a model, it computes:
- Distribution stats (mean, std, kurtosis, dynamic range)
- Sigma-based outlier counts (3σ, 5σ, 8σ, 10σ)
- Channel and input-dimension outlier detection
- INT4 quantization error (naive vs 99.9% clipped)

Then generates 9 chart types and a markdown report.

## Models Analyzed

| Model | Peak Kurtosis | Global \|max\| | INT4 Clip Improvement | Verdict |
|-------|-------------:|---------------:|----------------------:|---------|
| [GPT-2](models/gpt2/) | 790.5 | 17.10 | 90.8% | Severe outliers in MLP layers |
| [Qwen3.5-0.8B](models/qwen3.5-0.8b/) | 712.0 | 4.75 | 83.0% | Novel arch has new outlier patterns |
| [Llama 3.2 1B](models/llama-3.2-1b/) | 16.1 | 1.23 | 90.4% | Remarkably clean |

[Full comparison →](models/comparison.md)

## Pipeline

```
src/
├── run.py        → single entry point
├── analyze.py    → per-tensor stats (resumable JSONL)
├── visualize.py  → 9 chart types (dark theme)
└── report.py     → markdown report with embedded charts
```

**Resumable**: Analysis appends to JSONL line by line. If interrupted, rerun and it picks up where it left off.

**FP8 support**: Automatically dequantizes FP8 models (e.g., DeepSeek V3) using block-wise scaling.

## Output

```
results/<model>/
├── <model>_stats.jsonl              # Raw per-tensor statistics
├── <model>_weight_outlier_analysis.md
├── <model>_weight_outlier_analysis_summary.json
└── images/
    ├── 01_kurtosis_by_layer.png
    ├── 02_abs_max_by_layer.png
    ├── 03_dynamic_range_by_layer.png
    ├── 04_outlier_sigma_heatmap.png
    ├── 05_quant_error_by_layer.png
    ├── 06_outlier_dims.png
    ├── 07_component_summary.png
    ├── 08_worst_tensors.png
    └── 09_expert_heatmap.png (MoE only)
```

## Advanced Usage

```bash
# Skip analysis, regenerate charts + report
python src/run.py openai-community/gpt2 --skip-analyze

# Only regenerate report
python src/run.py openai-community/gpt2 --skip-analyze --skip-visualize

# Custom output directory
python src/run.py deepseek-ai/DeepSeek-V3 --output-dir ./ds_v3

# Run steps individually
python src/analyze.py --model openai-community/gpt2 --output stats.jsonl
python src/visualize.py --input stats.jsonl --output-dir charts/
python src/report.py --input stats.jsonl --images-dir charts/ --output report.md
```
