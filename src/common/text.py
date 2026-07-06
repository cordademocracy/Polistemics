from __future__ import annotations

import re

# Type alias for the compiled regex + placeholder mapping
NamePattern = tuple[re.Pattern[str], dict[str, str]]


def build_name_pattern(party_names: list[str]) -> NamePattern:
    """Build regex pattern and placeholder mapping for party name stripping."""
    sorted_names = sorted(party_names, key=lambda n: (-len(n), n.lower()))
    placeholder_map: dict[str, str] = {}
    for i, name in enumerate(sorted_names):
        letter = chr(ord("A") + i)
        placeholder_map[name.lower()] = f"Party {letter}"

    escaped = [re.escape(n) for n in sorted_names]
    pattern = re.compile("|".join(escaped), re.IGNORECASE)
    return pattern, placeholder_map


def strip_party_names(text: str, pattern: NamePattern) -> str:
    """Replace party name occurrences with neutral placeholders."""
    regex, placeholder_map = pattern

    def _replace(match: re.Match[str]) -> str:
        return placeholder_map[match.group(0).lower()]

    return regex.sub(_replace, text)
