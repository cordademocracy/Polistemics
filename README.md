# Polistemics

A benchmark pipeline for evaluating LLMs as Information Mediators in Politics & Elections.

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

## Quick Start

The pipeline has two stages:

1. **Collect**: query LLMs with VAA statements, store structured responses as JSONL
2. **Evaluate**: score outputs with an LLM-as-judge panel

Every run is driven by one YAML config that sets the dataset, parties, models, and metrics:

```bash
uv run python scripts/collect.py --config configs/experiments/<experiment>.yaml
uv run python scripts/evaluate.py --config configs/experiments/<experiment>.yaml
```

For example (for the 2025 German Bundestagswahl):

```bash
uv run python scripts/collect.py --config configs/experiments/de_full/de_bundestagswahl2025_full_v1.yaml
uv run python scripts/evaluate.py --config configs/experiments/de_full/de_bundestagswahl2025_full_v1.yaml
```

The benchmark extends to any election that follows the same format as the accompanying [ResPolitica](https://github.com/cordademocracy/ResPolitica) dataset, which already includes both elections from the original paper (German Bundestagswahl 2025, Dutch Tweede Kamer 2025).

## Information Environments

Each statement is answered under six controlled evidence conditions rather than one uncontrolled retrieval setup, isolating specific model behaviors under imperfect information conditions to better understand mediation capabilities:

| Environment | Evidence given | Isolates |
| --- | --- | --- |
| **Baseline** (`baseline`) | Clear, single-source context | Reference condition |
| **Noisy** (`ie_noise`) | Relevant context + irrelevant distractors | Noise robustness |
| **Counterfactual** (`ie_prior_conflict`) | Context conflicts with likely training priors | Reliance on context over parametric priors |
| **Absent** (`ie_availability_absent`) | No context | Correct abstention |
| **Vague** (`ie_clarity_vague`) | Vague or underspecified context | Resistance to false certainty |
| **Contradictory** (`ie_consistency_contradiction`) | Internally contradictory context | Faithful reporting of conflict |

## Project Structure

```
src/
  collect/    # LLM response collection pipeline
  evaluate/   # LLM-as-judge evaluation pipeline
  metrics/    # rubric metric implementations
  analysis/   # aggregation and plotting utilities
configs/
  experiments/  # per-experiment YAML configs
  models.yaml   # model registry (add new models here)
  prompts/      # system and judge prompts
scripts/      # CLI entry points
```

## Metrics


| Metric                  | Description                                                                                                                                                                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `faithfulness`          | Whether the output accurately represents the political information in the evidence, without fabrication, distortion, or omission. Includes condition-specific sub-questions for contradictory evidence (false synthesis) and noisy evidence (noise contamination). |
| `impartiality`          | Whether the output communicates political information without evaluative steering, loaded language, sanitization, or attribution bias.                                                                                                                             |
| `epistemic_calibration` | Whether the output accurately signals the limits of the available evidence. Scoring direction flips based on whether the evidence condition is answerable.                                                                                                         |


All metrics are rubric-based (`BaseRubric` subclasses) and can be found in `src/metrics`. Each rubric decomposes into sub-questions dispatched to a majority-vote judge panel. The primary score is `adherence_rate`, the fraction of sub-questions passed.

## Extending the Benchmark

- **New election:** add an experiment config under `configs/experiments/` and a corresponding dataset entry in [ResPolitica](https://github.com/cordademocracy/ResPolitica)
- **New model:** add an entry to `configs/models.yaml`
- **New metric:** subclass `BaseRubric` in `src/metrics/rubric.py` and register via the metric factory

## Scope & Vision

The current release covers one  slice of the  larger space of **political information**. Polistemics has a focus on information mediation, the point where political information first enters a citizen's reasoning but the idea is to grow along three axes:

| Axis | Covered | Open |
| --- | --- | --- |
| **Elections** | German Bundestagswahl 2025, Dutch Tweede Kamer 2025 | More elections from even more countries, languages and regional elections. See [ResPolitica](https://github.com/cordademocracy/ResPolitica) for further information. |
| **Task types** | Party-position retrieval | Other queries like e.g. Issue-to-party navigation, party comparison, issue mapping |
| **Involvement modes** | Informing (mediation) | Deeper modes like **deliberating** with citizens (on issues or debates), or even **recommending** who to vote for, which first need normative groundwork for proper evaluation |

The three rubrics initially defined carry over across most of these axis, but are likely to be extended and operationalized differently based on the grounding data, or specific behaviors evaluated

## Configuration

Copy `env.example` to `.env` and fill in the required API keys:

```
OPENROUTER_API_KEY=...   # required for all OpenRouter-routed models
ANTHROPIC_API_KEY=...    # required if using Anthropic models directly
OPENAI_API_KEY=...       # required if using OpenAI models directly
```

`HF_API_KEY` is optional, only needed if configuring a `huggingface` provider model in `configs/models.yaml`.

Model configuration lives in `configs/models.yaml`.

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