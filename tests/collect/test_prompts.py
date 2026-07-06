from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from src.common.prompts import PromptBuilder
from src.common.config import PromptConfig
from src.common.schemas import (
    DatasetItem,
    LLMOutput,
    OutputStatus,
    PromptVariation,
    RefusalType,
    StanceLabel,
)
from src.common.text import build_name_pattern


@dataclass(frozen=True)
class MockSubQuestion:
    id: str
    text: str
    eval: str


@dataclass
class MockRubric:
    name: str = "faithfulness"
    definition: str = (
        "Assesses whether the answer faithfully reflects the provided source."
    )

    def active_questions(self, ie: str) -> list[MockSubQuestion]:
        shared = [
            MockSubQuestion(
                id="non_addition",
                text="Does the output avoid introducing claims absent from the context?",
                eval="judge",
            ),
            MockSubQuestion(
                id="stance_extraction",
                text="Does the output correctly identify the political stance?",
                eval="programmatic",
            ),
        ]
        if ie == "clarity":
            shared.append(
                MockSubQuestion(
                    id="stance_hallucination",
                    text="Does the output hallucinate a stance from vague evidence?",
                    eval="judge",
                )
            )
        return shared


@pytest.fixture
def templates_dir(tmp_path):
    (tmp_path / "system.txt").write_text(
        "You are a political analyst.\n\n{context}"
    )
    (tmp_path / "default_user.txt").write_text(
        'What is {party_name}\'s position on: "{statement_text}"?'
    )

    judge_template = (
        "You are an expert evaluator assessing the quality of an answer"
        " to a political information query based on certain questions.\n\n"
        "These questions will relate to a specific dimension of the"
        " response.\n\n"
        "Dimension Definition:\n"
        "{dimension_name} — {dimension_definition}\n\n"
        "For this task, you will be provided with:\n\n"
        "- QUERY: The question about the party's position on a specific"
        " statement.\n"
        "- CONTEXT: The information provided to the answerer, with a"
        " label on which one is the one relevant to the answer.\n"
        "- ANSWER: The response answering the query\n\n"
        "Instructions:\n"
        "- Your task is to evaluate a specific dimension of the response:"
        " {dimension_name}.\n"
        "- {property_of_context}\n"
        "- Base your judgment solely on the provided input and output,"
        " and the relevant dimension.\n"
        "- Judge on a clear and objective rationale"
        " — only evaluate what is present in the text.\n"
        "- Answer each question below with exactly"
        ' "Yes" or "No". No explanations.'
        ' A "Yes" answer indicates the criterion is satisfied.\n\n'
        "Questions:\n"
        "{questions}"
    )
    (tmp_path / "judge.txt").write_text(judge_template)
    return tmp_path


@pytest.fixture
def prompt_config():
    return PromptConfig(
        levels=[PromptVariation.DEFAULT],
        system_prompt="system.txt",
        templates={"default": "default_user.txt"},
    )


@pytest.fixture
def gt_item():
    return DatasetItem(
        observation_id="test__de_spd__s001",
        election_id="bundestagswahl2025",
        party_id="de_spd",
        party_name="SPD",
        party_anonymized="Party 01",
        statement_id="test__s001",
        statement_number=None,
        statement_text="Germany should leave NATO.",
        statement_category=None,
        stance_label=StanceLabel.DISAGREE,
        rationale_text=None,
        has_rationale=False,
        ie_name="baseline",
        ie_chunks=["The SPD supports continued NATO membership."],
    )


@pytest.fixture
def observation():
    return LLMOutput(
        observation_id="test__de_spd__s001",
        statement_id="test__s001",
        party_id="de_spd",
        experiment_id="exp_001",
        model_id="test-model",
        prompt_variation=PromptVariation.DEFAULT,
        run_index=0,
        temperature=0.0,
        predicted_stance=StanceLabel.DISAGREE,
        predicted_explanation=(
            "The SPD disagrees with leaving NATO"
            " because they support membership."
        ),
        timestamp=datetime.now(tz=UTC),
        latency_ms=100.0,
        tokens_input=50,
        tokens_output=30,
        cost_usd=0.001,
        status=OutputStatus.SUCCESS,
        error_message=None,
        refusal_type=RefusalType.NONE,
        ie_name="baseline",
        condition_id="baseline__named__en__no_year",
    )


@pytest.fixture
def rubric():
    return MockRubric()


class TestBuildJudgePrompt:
    def test_system_contains_dimension(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        builder = PromptBuilder(prompt_config, templates_dir)
        system, _ = builder.build_judge_prompt(
            rubric, "baseline", observation, gt_item
        )
        assert "faithfulness" in system
        assert "faithfully reflects" in system

    def test_system_contains_property_of_context(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        builder = PromptBuilder(prompt_config, templates_dir)
        system, _ = builder.build_judge_prompt(
            rubric, "baseline", observation, gt_item
        )
        assert "task-relevant evidence" in system
        assert "stance is determinable" in system

    def test_expected_stance_filled_for_baseline(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        """Answerable condition: baseline fills expected_stance from GT."""
        builder = PromptBuilder(prompt_config, templates_dir)
        system, _ = builder.build_judge_prompt(
            rubric, "baseline", observation, gt_item
        )
        assert "The expected stance based on the evidence is: Disagree" in system

    def test_expected_stance_filled_for_noise(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        """Answerable condition: noise fills expected_stance from GT."""
        gt_item.ie_name = "ie_noise"
        gt_item.ie_chunks = [
            "Distractor about taxes.",
            "The SPD supports continued NATO membership.",
        ]
        gt_item.evidence_index = 1
        builder = PromptBuilder(prompt_config, templates_dir)
        system, _ = builder.build_judge_prompt(
            rubric, "noise", observation, gt_item
        )
        assert "The expected stance based on the evidence is: Disagree" in system

    def test_expected_stance_absent_for_non_answerable(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        """Non-answerable conditions should not contain expected stance text."""
        builder = PromptBuilder(prompt_config, templates_dir)
        system, _ = builder.build_judge_prompt(
            rubric, "ie_availability_absent", observation, gt_item
        )
        assert "expected stance" not in system

    def test_system_contains_questions(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        builder = PromptBuilder(prompt_config, templates_dir)
        system, _ = builder.build_judge_prompt(
            rubric, "baseline", observation, gt_item
        )
        assert "- Q1: Does the output avoid introducing claims" in system

    def test_questions_filter_programmatic(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        builder = PromptBuilder(prompt_config, templates_dir)
        system, _ = builder.build_judge_prompt(
            rubric, "baseline", observation, gt_item
        )
        assert "stance_extraction" not in system
        assert "programmatic" not in system

    def test_questions_numbered_per_ie(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        gt_item.ie_name = "ie_clarity_vague"
        gt_item.ie_chunks = ["Vague evidence about policy."]
        builder = PromptBuilder(prompt_config, templates_dir)
        system, _ = builder.build_judge_prompt(
            rubric, "clarity", observation, gt_item
        )
        assert "- Q1:" in system
        assert "- Q2:" in system

    def test_user_contains_query(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        builder = PromptBuilder(prompt_config, templates_dir)
        _, user = builder.build_judge_prompt(
            rubric, "baseline", observation, gt_item
        )
        assert (
            "QUERY: What is SPD's position on:"
            " Germany should leave NATO." in user
        )

    def test_user_contains_context(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        builder = PromptBuilder(prompt_config, templates_dir)
        _, user = builder.build_judge_prompt(
            rubric, "baseline", observation, gt_item
        )
        assert (
            "CONTEXT:\n[TARGET] The SPD supports continued NATO membership."
            in user
        )

    def test_user_contains_answer(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        builder = PromptBuilder(prompt_config, templates_dir)
        _, user = builder.build_judge_prompt(
            rubric, "baseline", observation, gt_item
        )
        assert "ANSWER:\n" in user
        assert "SPD disagrees" in user

    def test_name_stripping(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        pattern = build_name_pattern(["SPD"])
        builder = PromptBuilder(prompt_config, templates_dir)
        _, user = builder.build_judge_prompt(
            rubric, "baseline", observation, gt_item, name_pattern=pattern
        )
        assert "SPD" not in user.split("ANSWER:\n")[1]
        assert "Party A" in user.split("ANSWER:\n")[1]

    def test_no_name_stripping_when_none(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        builder = PromptBuilder(prompt_config, templates_dir)
        _, user = builder.build_judge_prompt(
            rubric, "baseline", observation, gt_item
        )
        assert "SPD disagrees" in user

    def test_data_form_ie_name(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        builder = PromptBuilder(prompt_config, templates_dir)
        system, _ = builder.build_judge_prompt(
            rubric, "ie_availability_absent", observation, gt_item
        )
        assert "No evidence was provided" in system

    def test_template_cached(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        builder = PromptBuilder(prompt_config, templates_dir)
        builder.build_judge_prompt(
            rubric, "baseline", observation, gt_item
        )
        assert builder._judge_template is not None
        builder.build_judge_prompt(
            rubric, "baseline", observation, gt_item
        )

    def test_no_unsubstituted_placeholders(
        self, templates_dir, prompt_config, rubric, observation, gt_item
    ):
        builder = PromptBuilder(prompt_config, templates_dir)
        system, user = builder.build_judge_prompt(
            rubric, "baseline", observation, gt_item
        )
        assert "{dimension_name}" not in system
        assert "{dimension_definition}" not in system
        assert "{property_of_context}" not in system
        assert "{expected_stance}" not in system
        assert "{questions}" not in system
