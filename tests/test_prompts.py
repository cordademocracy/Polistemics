
import pytest

from src.common.prompts import PromptBuilder
from src.common.config import PromptConfig
from src.common.schemas import DatasetItem, PromptVariation, StanceLabel


@pytest.fixture
def templates_dir(tmp_path):
    # System prompt uses {context} and {language} — populated by build_system_prompt(item).
    (tmp_path / "system.txt").write_text("You are a political analyst.\nRespond in {language}.\n\n{context}")
    (tmp_path / "default_user.txt").write_text('What is {party_name}\'s position on the following statement: "{statement_text}"?\nProvide their stance (Agree / Disagree / Neutral / Uncertain) and a brief answer.')
    return tmp_path


@pytest.fixture
def prompt_config():
    return PromptConfig(
        levels=[PromptVariation.DEFAULT],
        system_prompt="system.txt",
        templates={"default": "default_user.txt"},
    )


@pytest.fixture
def item():
    return DatasetItem(
        observation_id="test__de_spd__s001", election_id="bundestagswahl2025",
        party_id="de_spd", party_name="SPD", party_anonymized="Party 01",
        statement_id="test__s001",
        statement_number=None, statement_text="Deutschland soll aus der NATO austreten.",
        statement_category=None, stance_label=StanceLabel.DISAGREE,
        rationale_text=None, has_rationale=False,
        ie_name="baseline", ie_chunks=["The SPD supports NATO membership."],
    )


def test_build_system_prompt(templates_dir, prompt_config, item):
    builder = PromptBuilder(prompt_config, templates_dir, "German")
    system = builder.build_system_prompt(item)
    assert "political analyst" in system
    assert "SPD supports NATO" in system
    assert "German" in system


def test_build_system_prompt_english(templates_dir, prompt_config, item):
    builder = PromptBuilder(prompt_config, templates_dir, "English")
    system = builder.build_system_prompt(item)
    assert "English" in system


def test_build_system_prompt_missing_language_raises(templates_dir, prompt_config, item):
    builder = PromptBuilder(prompt_config, templates_dir)
    with pytest.raises(ValueError, match="response_language is not set"):
        builder.build_system_prompt(item)


def test_build_default_prompt(templates_dir, prompt_config, item):
    builder = PromptBuilder(prompt_config, templates_dir)
    prompt = builder.build_user_prompt(item, PromptVariation.DEFAULT)
    assert "SPD" in prompt
    assert "NATO" in prompt
    assert "{" not in prompt


@pytest.mark.skip(reason="context kwarg removed from build_user_prompt in new-data-format migration — context now injected via build_system_prompt(item)")
def test_build_contextual_with_architecture_context(templates_dir, prompt_config, item):
    pass
