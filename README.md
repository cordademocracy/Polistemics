# Polistemics

A benchmark pipeline for evaluating LLMs as Electoral Information Mediators.

## Overview

Polistemics is a diagnostic benchmark for evaluating LLMs as mediators of political information in elections. It measures whether models faithfully, impartially, and with calibrated confidence mediate political party positions under controlled information conditions. A cross-family LLM judge panel scores free-form model outputs on three rubric-based metrics (faithfulness, impartiality, epistemic calibration) via majority vote. Experiments cover the 2025 German Bundestag and Dutch Tweede Kamer elections. All elections, models, and metrics are config-driven and extensible without code changes.

## Installation

```bash
git clone <repo-url>
cd Polistemics
uv venv && uv sync --all-extras
cp env.example .env  # fill in API keys
```

Requires Python 3.12.

## Quick Start, Reproduce Paper Experiments

The pipeline has two stages:

1. **Collect**: query LLMs with VAA statements, store structured responses as JSONL
2. **Evaluate**: score outputs with an LLM-as-judge panel; produces the tidy scores table automatically

### German Bundestagswahl 2025

```bash
uv run python scripts/collect.py --config configs/experiments/de_full/de_bundestagswahl2025_full_v1.yaml
uv run python scripts/evaluate.py --config configs/experiments/de_full/de_bundestagswahl2025_full_v1.yaml
```

### Dutch Tweede Kamer 2025

```bash
uv run python scripts/collect.py --config configs/experiments/nl_full/nl_tk2025_full_v1.yaml
uv run python scripts/evaluate.py --config configs/experiments/nl_full/nl_tk2025_full_v1.yaml
```

## Project Structure

```
src/
  collect/    # LLM response collection pipeline
  evaluate/   # LLM-as-judge evaluation pipeline
  metrics/    # fairness metric implementations
  analysis/   # aggregation and plotting utilities
configs/
  experiments/  # per-experiment YAML configs
  models.yaml   # model registry (add new models here)
  prompts/      # system and judge prompts
scripts/      # CLI entry points
tests/        # 45+ unit and integration tests
```

## Metrics


| Metric                  | Description                                                                                                                                                                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `faithfulness`          | Whether the output accurately represents the political information in the evidence, without fabrication, distortion, or omission. Includes condition-specific sub-questions for contradictory evidence (false synthesis) and noisy evidence (noise contamination). |
| `impartiality`          | Whether the output communicates political information without evaluative steering, loaded language, sanitization, or attribution bias.                                                                                                                             |
| `epistemic_calibration` | Whether the output accurately signals the limits of the available evidence. Scoring direction flips based on whether the evidence condition is answerable.                                                                                                         |


All metrics are rubric-based (`BaseRubric` subclasses). Each rubric decomposes into sub-questions dispatched to a majority-vote judge panel. The primary score is `adherence_rate`, the fraction of sub-questions passed.

## Extending the Benchmark

- **New election:** add an experiment config under `configs/experiments/` and a corresponding dataset entry in [ResPolitica](https://github.com/cordademocracy/ResPolitica)
- **New model:** add an entry to `configs/models.yaml`; no code changes required
- **New metric:** subclass `BaseRubric` in `src/metrics/rubric.py` and register via the metric factory

## Configuration

Copy `env.example` to `.env` and fill in the required API keys:

```
OPENROUTER_API_KEY=...   # required for all OpenRouter-routed models
ANTHROPIC_API_KEY=...    # required if using Anthropic models directly
OPENAI_API_KEY=...       # required if using OpenAI models directly
```

`HF_API_KEY` is optional, only needed if configuring a `huggingface` provider model in `configs/models.yaml`.

Model configuration lives in `configs/models.yaml`. No model names are hardcoded in source.

## Tests

```bash
uv run pytest tests/ -q
```

## Citation

```bibtex
@misc{polistemics2026,
  title   = {Polistemics: Evaluating LLMs as Information Mediators in Politics \& Elections},
  author  = {Peters, Baran},
  year    = {2026},
  note    = {Preprint}
}
```

## License

See [LICENSE](LICENSE).