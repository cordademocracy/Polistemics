# Synthetic E2E Test Dataset

> Used in: `tests/test_e2e_synthetic.py`
> Purpose: Validate full pipeline (collection → evaluation → scoring) with known edge cases

## Dataset Shape

- **4 parties:** CDU/CSU, SPD, GRÜNE, AfD
- **4 statements:** S01 (Security), S03 (Migration), S04 (Environment), S05 (Social)
- **16 observations** (4 × 4), **K=3 runs each** = **48 outputs total**
- **Model:** model_a, prompt_variation: minimal, temperature: 0.3

## Edge Cases Covered

| # | Case | Where | What it tests |
|---|------|-------|---------------|
| 1 | Correct stance, all 3 classes | CDU S01 (Agree), SPD S01 (Neutral), GRÜNE S01 (Disagree) | Stance accuracy happy path |
| 2 | Wrong stance, majority vote 2:1 | CDU S03 (predicts Agree, GT is Neutral) | Majority vote with disagreement |
| 3 | Wrong stance, three-way tie | AfD S05 (Agree/Neutral/Disagree, GT is Neutral) | Tie-breaking in aggregate |
| 4 | K=3 identical runs | Most cases | agreement_rate = 1.0 |
| 5 | Highly neutral explanation | SPD S03 ("lehnt ab... verweist auf") | Sentiment: high P(neu) expected |
| 6 | Strongly negative framing | AfD S03 ("katastrophal", "unkontrolliert") | Sentiment: negative compound |
| 7 | Strongly positive framing | GRÜNE S04 ("leidenschaftlich", "historische Chance") | Sentiment: positive compound |
| 8 | Mixed/balanced sentiment | CDU S05 ("befürwortet... aber problematisch") | Sentiment: near-zero compound |
| 9 | Multiple party names in text | AfD S01 ("Anders als SPD und GRÜNE...") | Name-stripping of non-focal parties |
| 10 | Immigration topic for AfD | AfD S03 | Face validity: expected negative signal |

## Real Model Observations (2026-04-07)

Ran real mDeBERTa inference on these cases. Key findings:

| Party | Topic | Expected | pos | neg | neu | compound | Notes |
|-------|-------|----------|-----|-----|-----|----------|-------|
| CDU/CSU | Security | Neutral/factual | 0.984 | 0.006 | 0.009 | +0.978 | "befürwortet" drives positive — content/style conflation |
| AfD | Migration | Strongly negative | 0.018 | 0.961 | 0.021 | -0.943 | Model detects negative framing clearly |
| GRÜNE | Environment | Strongly positive | 0.990 | 0.004 | 0.006 | +0.986 | "leidenschaftlich" drives positive |
| SPD | Migration | Factual/mild | 0.044 | 0.160 | 0.796 | -0.117 | Only genuinely neutral result |

**Content/style conflation observed:** The model conflates content polarity (what the party supports/opposes) with evaluative tone (how the text treats the party). "befürwortet" is factually descriptive but the model reads it as positive sentiment toward the party. This is the known limitation — to be investigated further with real pilot outputs and hypothesis template sensitivity analysis.

## Extending This Dataset

To add new test cases:
1. Add entries to `GT_STANCES` dict (party_id, statement_id) → StanceLabel
2. Add entries to `EXPLANATIONS` dict with K=3 runs as list of (stance, text) tuples
3. If adding new parties: extend `PARTIES` list
4. If adding new statements: extend `STATEMENTS` list
5. Run `pytest tests/test_e2e_synthetic.py -v` to verify
