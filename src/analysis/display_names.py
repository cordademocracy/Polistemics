"""Shared display-name mappings for analysis notebooks."""

from __future__ import annotations

from src.analysis.plotting import DisplayNames

DISPLAY_NAMES = DisplayNames(
    party={
        "de_cdu___csu": "CDU/CSU",
        "de_spd": "SPD",
        "de_grüne": "Grüne",
        "de_fdp": "FDP",
        "de_afd": "AfD",
        "de_die_linke": "Linke",
        "de_bsw": "BSW",
        "nl_vvd": "VVD",
        "nl_d66": "D66",
        "nl_cda": "CDA",
        "nl_pvv": "PVV",
        "nl_groenlinks-pvda": "GL-PvdA",
        "nl_sp": "SP",
        "nl_bbb": "BBB",
        "nl_ja21": "JA21",
    },
    model={
        "claude_sonnet_4_6": "Claude",
        "gpt_5_4": "GPT",
        "qwen3_6_flash": "Qwen",
    },
    ie={
        "baseline": "Baseline",
        "availability": "Absent",
        "clarity": "Vague",
        "consistency": "Contradictory",
        "noise": "Noisy",
        "prior_conflict": "Counterfactual",
    },
    rubric={
        "faithfulness": "Faithfulness",
        "impartiality": "Impartiality",
        "epistemic_calibration": "Epistemic calibration",
    },
    # Short "IDdescriptor" labels for compact SQ axes. IDs follow the
    # metric-definition order in src/metrics/: F* faithfulness, E* epistemic
    # calibration, I* impartiality.
    subquestion={
        "position_representation": "F1 Position",
        "information_fabrication": "F2 Fabrication",
        "false_synthesis": "F3 False-synth",
        "noise_contamination": "F4 Noise",
        "epistemic_certainty": "EC1 Certainty",
        "epistemic_hedging": "EC2 Hedging",
        "context_transparency": "EC3 Transparency",
        "parametric_fallback": "EC4 Fallback",
        "endorsement": "I1 Endorsement",
        "condemnation": "I2 Condemnation",
        "loaded_language": "I3 Loaded-lang",
        "sanitization": "I4 Sanitization",
        "attribution_bias": "I5 Attribution",
    },
)

# Fixed per-party colours so a party reads as the same colour in every plot.
# Conventional German party colours, nudged for contrast on a white ground.
# NOTE: SPD red / Grüne green are NOT deuteranopia-safe — this is a deliberate
# trade-off (party-identity colours give instant recognition in political plots).
# Non-party plots keep the colourblind palette from set_analysis_theme().
PARTY_COLORS: dict[str, str] = {
    "de_cdu___csu": "#2B2B2B",
    "de_spd": "#E3000F",
    "de_grüne": "#1FA12A",
    "de_fdp": "#E8B800",
    "de_afd": "#0A8FCB",
    "de_die_linke": "#BE3075",
    "de_bsw": "#8E44AD",
    "nl_vvd": "#0065A4",
    "nl_d66": "#00A95D",
    "nl_cda": "#007B5F",
    "nl_pvv": "#003082",
    "nl_groenlinks-pvda": "#006C2A",
    "nl_sp": "#E1001A",
    "nl_bbb": "#8B5E3C",
    "nl_ja21": "#1A1A2E",
}

# Canonical model ordering (by overall benchmark rank) and fixed accent colours,
# reused across every figure so a model reads identically everywhere. Brand-
# inspired trio; also serves as the colour hook for model logos/icons. Keyed by
# raw model ID (matches the DISPLAY_NAMES.model map above).
MODEL_ORDER: list[str] = [
    "claude_sonnet_4_6",
    "gpt_5_4",
    "qwen3_6_flash",
]

MODEL_COLORS: dict[str, str] = {
    "claude_sonnet_4_6": "#D97757",  # Claude — rust/coral
    "gpt_5_4": "#111111",            # GPT — black
    "qwen3_6_flash": "#615CED",      # Qwen — purple
}


# Canonical left-to-right ordering for party axes/legends (raw IDs).
PARTY_ORDER: list[str] = [
    "de_cdu___csu",
    "de_spd",
    "de_grüne",
    "de_fdp",
    "de_afd",
    "de_die_linke",
    "de_bsw",
    "nl_vvd",
    "nl_d66",
    "nl_cda",
    "nl_pvv",
    "nl_groenlinks-pvda",
    "nl_sp",
    "nl_bbb",
    "nl_ja21",
]


# Canonical left-to-right SQ ordering, grouped by rubric then metric-definition
# order. Drives column order + rubric-block separators in the disparity heatmap.
SUBQUESTION_ORDER: list[str] = [
    # faithfulness
    "position_representation",
    "information_fabrication",
    "false_synthesis",
    "noise_contamination",
    # epistemic_calibration
    "epistemic_certainty",
    "epistemic_hedging",
    "context_transparency",
    "parametric_fallback",
    # impartiality
    "endorsement",
    "condemnation",
    "loaded_language",
    "sanitization",
    "attribution_bias",
]

# Canonical rubric ordering (faithfulness → epistemic calibration → impartiality),
# matching the F/E/I sub-question grouping. Drives column order on model×rubric
# leaderboards and any rubric-keyed axis.
RUBRIC_ORDER: list[str] = [
    "faithfulness",
    "epistemic_calibration",
    "impartiality",
]


# Compact rubric codes for space-tight axes (parallel to the F1..I5 subquestion
# codes). F = faithfulness, EC = epistemic calibration, I = impartiality. Decode
# once in the methods / first-use caption, then reuse without re-explaining.
RUBRIC_CODES: dict[str, str] = {
    "faithfulness": "F",
    "epistemic_calibration": "EC",
    "impartiality": "I",
}


# IE grouping by evidence conclusiveness: interfering conditions retain conclusive
# evidence; inconclusive conditions weaken or remove it. Drives row/column order
# and group separators. Edit here to regroup.
IE_ROW_GROUPS: list[tuple[str, list[str]]] = [
    ("Interfering", ["baseline", "noise", "prior_conflict"]),
    ("Inconclusive", ["availability", "clarity", "consistency"]),
]


# SQ -> rubric, used to draw rubric-block separators on SQ-keyed plots.
SUBQUESTION_RUBRIC: dict[str, str] = {
    "position_representation": "faithfulness",
    "information_fabrication": "faithfulness",
    "false_synthesis": "faithfulness",
    "noise_contamination": "faithfulness",
    "epistemic_certainty": "epistemic_calibration",
    "epistemic_hedging": "epistemic_calibration",
    "context_transparency": "epistemic_calibration",
    "parametric_fallback": "epistemic_calibration",
    "endorsement": "impartiality",
    "condemnation": "impartiality",
    "loaded_language": "impartiality",
    "sanitization": "impartiality",
    "attribution_bias": "impartiality",
}


def party_palette(names: DisplayNames = DISPLAY_NAMES) -> dict[str, str]:
    """Map party *display* labels to their fixed colours.

    Pass the result straight to seaborn ``palette=`` when a plot's party column
    holds display labels (the usual case after ``names.apply(raw, "party")``).

    Args:
        names: Display-name registry used to resolve raw party IDs to labels.

    Returns:
        Mapping from party display label to hex colour.
    """

    return {names.apply(raw, "party"): color for raw, color in PARTY_COLORS.items()}


def model_palette(names: DisplayNames = DISPLAY_NAMES) -> dict[str, str]:
    """Map model *display* labels to their fixed brand colours.

    Pass straight to seaborn ``palette=`` when a plot's model column holds
    display labels (the usual case after ``names.apply(raw, "model")``).

    Args:
        names: Display-name registry used to resolve raw model IDs to labels.

    Returns:
        Mapping from model display label to hex colour.
    """

    return {names.apply(raw, "model"): color for raw, color in MODEL_COLORS.items()}

