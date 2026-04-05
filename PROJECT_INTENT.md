# Project Intent

This repo is for deep, model-specific analysis of model weights.

The goal is not to support many models with a single generic pipeline. The goal is to do a good job on each model, even if that means each model keeps a substantial amount of specialized code.

## Core Structure

- Each analyzed model should live in its own folder.
- Model folders should be grouped under a higher-level model family folder.
- The target layout is model-family first, then model-specific implementation.

Example target layout:

```text
models/
  openai/
    gpt2/
  llama/
    llama-3.2-1b/
  qwen/
    qwen3.5-0.8b/
shared/
```

This is a target direction for the repo structure. The current layout may remain in transition until it is worth reorganizing.

## What Each Model Folder Should Contain

Each model folder should contain enough code and outputs to fully support that model's analysis workflow.

For each model, the code should be able to:

1. Access the model weights.
2. Load the model either fully or layer by layer when disk or memory limits require it.
3. Run the model-specific analysis logic.
4. Produce machine-readable data artifacts.
5. Produce visual artifacts.
6. Produce a final Markdown report that presents the findings and links to the generated images.

In practice, a model folder will usually contain:

- Loading and traversal code for that model.
- Analysis code for that model.
- Generated data files such as JSON, JSONL, or other intermediate artifacts.
- Generated images and plots.
- A final report in Markdown.

## Shared Code

Reusable code can live in `shared/`, but only when it is genuinely shared.

Examples of good shared code:

- Basic tensor I/O helpers.
- Common statistics utilities.
- Plotting helpers or styling helpers.
- Report-generation utilities that are still generic.

However, I expect most real analysis logic to stay model-specific.

That includes things like:

- Tensor selection rules.
- Architecture-aware classification.
- Model-specific visualizations.
- Model-specific interpretation.
- Model-specific narrative in the final report.

The repo should prefer some duplication over premature abstraction when the duplication keeps the analysis easier to understand and easier to extend for a specific model.

## Working Philosophy

- Organize around models, not around a universal framework.
- Optimize for depth on one model at a time.
- Keep shared code small and useful.
- Let each model have specialized code when the analysis demands it.
- Treat the report, data artifacts, and visual artifacts as first-class outputs of each model folder.
