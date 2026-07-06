# src/metrics

Defines the metric interface (`BaseMetric`), concrete metric implementations, and
the registry that maps metric name strings to metric classes.

## BaseMetric interface contract

Defined in `base.py`. All metrics must implement:

| Method | Signature | When called |
|--------|-----------|-------------|
| `score(output, ground_truth)` | `(LLMOutput, DatasetItem) → dict[str, float]` | Per individual output, by the evaluation pipeline |
| `aggregate(run_scores)` | `(list[dict[str, float]]) → dict[str, float]` | After K runs for the same (obs, model, pv, temp), by the pipeline |
| `summarize(batch_scores)` | `(list[dict[str, float]]) → dict[str, float]` | On a batch of aggregated scores (e.g., one party), by the analysis module |

Two required properties: `name: str` (registry key, used in file paths) and
`aggregation_method: str` (human-readable description stored on `AggregatedScore`).

**Contract:** `aggregate()` must preserve all score keys that `summarize()` needs.

## Adding a new metric

1. Create `src/metrics/{your_metric}.py`, subclass `BaseMetric`
2. Implement `name`, `aggregation_method`, `score()`, `aggregate()`, `summarize()`
3. Register it in `registry.py`:
   ```python
   from src.metrics.your_metric import YourMetric
   METRIC_REGISTRY["your_metric_name"] = YourMetric
   ```
4. Add the name to the metrics list in the relevant experiment config YAML

## Registry

`registry.py` holds `METRIC_REGISTRY: dict[str, type[BaseMetric]]`. The evaluation
pipeline uses this dict to instantiate metrics by name — no pipeline changes needed
when adding new metrics.
