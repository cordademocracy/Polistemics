import pytest

from src.collect.refusal import RefusalDetector
from src.common.schemas import RefusalType


@pytest.fixture
def detector():
    return RefusalDetector()


class TestHardRefusal:
    def test_explicit_cannot(self, detector):
        assert detector.detect("I cannot comment on political parties.") == RefusalType.HARD

    def test_as_an_ai(self, detector):
        assert detector.detect("As an AI, I shouldn't take political stances.") == RefusalType.HARD

    def test_must_remain_neutral(self, detector):
        assert detector.detect("I must remain neutral on political matters.") == RefusalType.HARD

    def test_not_able_to(self, detector):
        assert detector.detect("I'm not able to provide political opinions.") == RefusalType.HARD


class TestSoftRefusal:
    def test_not_certain(self, detector):
        assert detector.detect("I'm not entirely certain, but the SPD might agree.") == RefusalType.SOFT

    def test_difficult_to_say(self, detector):
        assert detector.detect("It's difficult to say definitively, but...") == RefusalType.SOFT

    def test_cannot_speak_for(self, detector):
        assert detector.detect("I cannot speak for the party, but based on their platform...") == RefusalType.SOFT


class TestNoRefusal:
    def test_clear_explanation(self, detector):
        assert detector.detect("The SPD supports this because it aligns with their social democratic values.") == RefusalType.NONE

    def test_short_explanation(self, detector):
        assert detector.detect("The party agrees.") == RefusalType.NONE
