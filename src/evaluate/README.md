# src/evaluate

Orchestrates metric computation on collected experiment data: loads outputs, runs
selected metrics, aggregates K runs, and persists scores per metric.

## Data flow

```
experiment_dir/
  outputs.jsonl        → load_outputs_from_path()  → list[LLMOutput]
  ground_truth.jsonl   → load_ground_truth()        → list[DatasetItem]
        |
        ↓ (optional) filter_outputs()
        |
        ↓ for each metric in metric_names
        |
        ↓ BaseMetric.score(output, gt)   → ItemScore per output
        ↓ BaseMetric.aggregate(k_scores) → AggregatedScore per (obs, model, pv, temp)
        |
        ↓ persist to:
  scores/{metric_name}/item_scores.jsonl
  scores/{metric_name}/aggregated_scores.jsonl
```

## Pipeline steps

1. Load `outputs.jsonl` + `ground_truth.jsonl` from the experiment directory
2. Optionally subset with `OutputFilter` (party, model, prompt_variation, temperature, status)
3. Build `{observation_id → DatasetItem}` lookup
4. For each requested metric (skips if scores exist unless `force=True`):
   - Instantiate metric from `METRIC_REGISTRY`
   - Call `metric.score(output, gt)` per successful output → `ItemScore`
   - Group by `(observation_id, model_id, prompt_variation, temperature)` → call `metric.aggregate()` → `AggregatedScore`
   - Write both to `scores/{metric_name}/`
5. Return `EvaluationResult` with all metric results

## Metrics plug in via BaseMetric

Metrics are referenced by name string. `EvaluationPipeline` looks up the name in
`METRIC_REGISTRY` and instantiates the class. No changes to the pipeline needed
when adding new metrics — only the registry entry is required.

See `src/metrics/base.py` for the interface contract.

## Running

```bash
uv run python scripts/evaluate.py --config configs/experiments/de_full/de_bundestagswahl2025_full_v1.yaml
uv run python scripts/evaluate.py --config configs/experiments/de_full/de_bundestagswahl2025_full_v1.yaml --metrics faithfulness
uv run python scripts/evaluate.py --config configs/experiments/de_full/de_bundestagswahl2025_full_v1.yaml --filter-parties de_cdu,de_afd --force
```
