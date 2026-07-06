from __future__ import annotations

import re

from src.common.schemas import RefusalType

HARD_REFUSAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"I cannot.*comment.*politic", re.IGNORECASE),
    re.compile(r"as an AI.*I (cannot|shouldn't|don't|should not)", re.IGNORECASE),
    re.compile(r"I'm not able to.*politic", re.IGNORECASE),
    re.compile(r"I must remain neutral", re.IGNORECASE),
    re.compile(r"I (cannot|can't|shouldn't|should not).*take.*stance", re.IGNORECASE),
    re.compile(
        r"I (cannot|can't|shouldn't|should not).*provide.*political.*opinion",
        re.IGNORECASE,
    ),
    re.compile(r"not appropriate for me to", re.IGNORECASE),
]

SOFT_REFUSAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"I'm not (entirely )?certain", re.IGNORECASE),
    re.compile(r"I cannot speak for", re.IGNORECASE),
    re.compile(r"it's (difficult|hard) (for me )?(to say|to determine)", re.IGNORECASE),
    re.compile(r"I (can't|cannot) (definitively|conclusively)", re.IGNORECASE),
    re.compile(r"I'?m not (sure|confident) (enough )?to", re.IGNORECASE),
]


class RefusalDetector:
    """Post-hoc refusal detection via pattern matching on explanation text."""

    def detect(self, explanation: str) -> RefusalType:
        for pattern in HARD_REFUSAL_PATTERNS:
            if pattern.search(explanation):
                return RefusalType.HARD
        for pattern in SOFT_REFUSAL_PATTERNS:
            if pattern.search(explanation):
                return RefusalType.SOFT
        return RefusalType.NONE
