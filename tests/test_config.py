from src.common.config import (
    CollectionConfig,
    DatasetFilter,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PromptConfig,
    RunConfig,
    WebSearchConfig,
    load_experiment_config,
    load_model_configs,
)
from src.common.schemas import PromptVariation


def test_model_config_creation():
    mc = ModelConfig(id="model_a", provider="openrouter", model="placeholder",
                     max_tokens=1024, api_key_env="OPENROUTER_API_KEY")
    assert mc.provider == "openrouter"
    assert mc.max_tokens == 1024
    assert mc.reasoning_effort is None


def test_model_config_with_reasoning_effort():
    mc = ModelConfig(
        id="gpt_5_4", provider="openrouter", model="openai/gpt-5.4",
        max_tokens=1024, api_key_env="OPENROUTER_API_KEY",
        reasoning_effort="medium",
    )
    assert mc.reasoning_effort == "medium"


def test_model_config_reasoning_effort_default_none():
    mc = ModelConfig(id="x", provider="openai", model="m")
    assert mc.reasoning_effort is None


def test_experiment_config_creation():
    config = ExperimentConfig(
        experiment_id="test_v1", experiment_name="Test",
        dataset=DatasetFilter(source="ResPolitica", election_ids=["bundestagswahl2025"]),
        model_ids=["model_a"],
        prompts=PromptConfig(levels=[PromptVariation.MINIMAL], system_prompt="system.txt",
                             templates={"minimal": "minimal_user.txt"}),
        runs=[RunConfig(temperature=0.0, k=1)],
        metrics=["faithfulness"],
    )
    assert len(config.model_ids) == 1
    assert config.conditions.ie_levels == "all"
    assert config.conditions.party_label == "real"


def test_collection_config_defaults():
    cc = CollectionConfig()
    assert cc.max_retries == 3
    assert cc.concurrency.default == 5


def test_load_model_configs(tmp_path):
    yaml_content = 'models:\n  - id: "model_a"\n    provider: "openrouter"\n    model: "placeholder"\n    max_tokens: 1024\n    api_key_env: "OPENROUTER_API_KEY"\n  - id: "model_b"\n    provider: "anthropic"\n    model: "placeholder"\n    max_tokens: 512\n    api_key_env: "ANTHROPIC_API_KEY"\n'
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text(yaml_content)
    configs = load_model_configs(str(yaml_file))
    assert "model_a" in configs
    assert "model_b" in configs
    assert configs["model_b"].max_tokens == 512


def test_load_model_configs_with_reasoning_effort(tmp_path):
    yaml_content = (
        'models:\n'
        '  - id: "gpt_5_4"\n'
        '    provider: "openrouter"\n'
        '    model: "openai/gpt-5.4"\n'
        '    max_tokens: 1024\n'
        '    api_key_env: "OPENROUTER_API_KEY"\n'
        '    reasoning_effort: "medium"\n'
    )
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text(yaml_content)
    configs = load_model_configs(str(yaml_file))
    assert configs["gpt_5_4"].reasoning_effort == "medium"


def test_load_experiment_config(tmp_path):
    yaml_content = '''experiment_id: "test_v1"
experiment_name: "Test"
dataset:
  source: "ResPolitica"
  election_ids: ["bundestagswahl2025"]
  party_filter: ["SPD"]
model_ids: ["model_a"]
prompts:
  levels: ["minimal"]
  system_prompt: "system.txt"
  templates:
    minimal: "minimal_user.txt"
runs:
  - temperature: 0.0
    k: 1
metrics: ["faithfulness"]
collection:
  max_retries: 2
  concurrency:
    default: 3
    per_model:
      model_a: 5
'''
    yaml_file = tmp_path / "experiment.yaml"
    yaml_file.write_text(yaml_content)
    config = load_experiment_config(str(yaml_file))
    assert config.experiment_id == "test_v1"
    assert config.collection.max_retries == 2
    assert config.collection.concurrency.per_model["model_a"] == 5


def test_dataset_filter_with_path():
    df = DatasetFilter(
        path="datasets/respolitica_unified_federal.parquet",
        election_ids=["bundestagswahl2025"],
        party_filter=["SPD"],
    )
    assert df.path == "datasets/respolitica_unified_federal.parquet"
    assert df.source == "ResPolitica"


def test_evaluation_config_defaults():
    ec = EvaluationConfig()
    assert ec.bootstrap_resamples == 10000
    assert ec.significance_threshold == 0.05


def test_experiment_config_with_evaluation():
    config = ExperimentConfig(
        experiment_id="test",
        experiment_name="Test",
        dataset=DatasetFilter(
            path="datasets/test.parquet",
            election_ids=["test2025"],
        ),
        model_ids=["model_a"],
        prompts=PromptConfig(
            levels=["minimal"],
            system_prompt="system.txt",
            templates={"minimal": "min.txt"},
        ),
        runs=[RunConfig(temperature=0.0, k=1)],
        metrics=["faithfulness"],
    )
    assert config.evaluation.bootstrap_resamples == 10000
    assert config.dataset.path == "datasets/test.parquet"


def test_model_config_api_key_env_as_list():
    mc = ModelConfig(
        id="rotated", provider="openrouter", model="placeholder",
        api_key_env=["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"],
    )
    assert mc.api_key_env == ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"]
    assert isinstance(mc.api_key_env, list)


def test_model_config_api_key_env_as_string_still_works():
    mc = ModelConfig(
        id="single", provider="openai", model="placeholder",
        api_key_env="OPENAI_API_KEY",
    )
    assert mc.api_key_env == "OPENAI_API_KEY"
    assert isinstance(mc.api_key_env, str)


def test_model_config_openrouter_provider():
    mc = ModelConfig(
        id="claude_sonnet", provider="openrouter", model="anthropic/claude-sonnet-4.6",
        max_tokens=1024, api_key_env="OPENROUTER_API_KEY",
        openrouter_provider={"order": ["amazon-bedrock"], "allow_fallbacks": True},
    )
    assert mc.openrouter_provider == {"order": ["amazon-bedrock"], "allow_fallbacks": True}


def test_model_config_openrouter_provider_default_none():
    mc = ModelConfig(id="x", provider="openrouter", model="m")
    assert mc.openrouter_provider is None


def test_load_model_configs_with_openrouter_provider(tmp_path):
    yaml_content = (
        "models:\n"
        '  - id: "routed_model"\n'
        '    provider: "openrouter"\n'
        '    model: "anthropic/claude-sonnet-4.6"\n'
        "    max_tokens: 1024\n"
        '    api_key_env: "OPENROUTER_API_KEY"\n'
        "    openrouter_provider:\n"
        '      order: ["amazon-bedrock"]\n'
        "      allow_fallbacks: true\n"
    )
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text(yaml_content)
    configs = load_model_configs(str(yaml_file))
    assert configs["routed_model"].openrouter_provider == {
        "order": ["amazon-bedrock"],
        "allow_fallbacks": True,
    }


def test_load_model_configs_with_key_rotation(tmp_path):
    yaml_content = (
        "models:\n"
        '  - id: "rotated_model"\n'
        '    provider: "openrouter"\n'
        '    model: "placeholder"\n'
        "    max_tokens: 1024\n"
        "    api_key_env:\n"
        '      - "OPENROUTER_API_KEY"\n'
        '      - "OPENROUTER_API_KEY_2"\n'
    )
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text(yaml_content)
    configs = load_model_configs(str(yaml_file))
    assert "rotated_model" in configs
    assert configs["rotated_model"].api_key_env == [
        "OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY_2",
    ]


def test_web_search_config_defaults():
    ws = WebSearchConfig()
    assert ws.enabled is False
    assert ws.search_context_size == "medium"


def test_web_search_config_custom():
    ws = WebSearchConfig(enabled=True, search_context_size="high")
    assert ws.enabled is True
    assert ws.search_context_size == "high"


def test_model_config_with_web_search():
    mc = ModelConfig(
        id="search_model", provider="openrouter", model="placeholder",
        web_search=WebSearchConfig(enabled=True, search_context_size="low"),
    )
    assert mc.web_search is not None
    assert mc.web_search.enabled is True
    assert mc.web_search.search_context_size == "low"


def test_model_config_web_search_default_none():
    mc = ModelConfig(id="x", provider="openrouter", model="m")
    assert mc.web_search is None


def test_load_model_configs_with_web_search(tmp_path):
    yaml_content = (
        "models:\n"
        '  - id: "search_model"\n'
        '    provider: "openrouter"\n'
        '    model: "placeholder"\n'
        "    max_tokens: 1024\n"
        '    api_key_env: "OPENROUTER_API_KEY"\n'
        "    web_search:\n"
        "      enabled: true\n"
        '      search_context_size: "high"\n'
    )
    yaml_file = tmp_path / "models.yaml"
    yaml_file.write_text(yaml_content)
    configs = load_model_configs(str(yaml_file))
    assert configs["search_model"].web_search is not None
    assert configs["search_model"].web_search.enabled is True
    assert configs["search_model"].web_search.search_context_size == "high"


# ---------------------------------------------------------------------------
# JudgesConfig.timeout_seconds
# ---------------------------------------------------------------------------


def test_judges_config_timeout_seconds_defaults_to_none() -> None:
    """timeout_seconds is None when absent from config."""
    from src.common.config import JudgesConfig
    config = JudgesConfig()
    assert config.timeout_seconds is None


def test_judges_config_timeout_seconds_parses_float() -> None:
    """timeout_seconds parses correctly when provided as a float."""
    from src.common.config import JudgesConfig
    config = JudgesConfig(timeout_seconds=30.0)
    assert config.timeout_seconds == 30.0


def test_judges_config_timeout_seconds_from_yaml(tmp_path) -> None:
    """timeout_seconds parses correctly from a YAML experiment config."""
    yaml_content = '''experiment_id: "timeout_test"
experiment_name: "Timeout Test"
dataset:
  election_ids: ["bw2025"]
model_ids: ["model_a"]
prompts:
  levels: ["minimal"]
  system_prompt: "system.txt"
  templates:
    minimal: "minimal_user.txt"
runs:
  - temperature: 0.0
    k: 1
metrics: ["faithfulness"]
judges:
  panel: ["openai:gpt-4o-mini"]
  temperature: 0.0
  timeout_seconds: 45.0
'''
    yaml_file = tmp_path / "experiment.yaml"
    yaml_file.write_text(yaml_content)
    config = load_experiment_config(str(yaml_file))
    assert config.judges.timeout_seconds == 45.0


def test_judges_config_timeout_seconds_absent_from_yaml_is_none(tmp_path) -> None:
    """timeout_seconds is None when the YAML judges block omits it."""
    yaml_content = '''experiment_id: "no_timeout"
experiment_name: "No Timeout"
dataset:
  election_ids: ["bw2025"]
model_ids: ["model_a"]
prompts:
  levels: ["minimal"]
  system_prompt: "system.txt"
  templates:
    minimal: "minimal_user.txt"
runs:
  - temperature: 0.0
    k: 1
metrics: ["faithfulness"]
judges:
  panel: ["openai:gpt-4o-mini"]
  temperature: 0.0
'''
    yaml_file = tmp_path / "experiment.yaml"
    yaml_file.write_text(yaml_content)
    config = load_experiment_config(str(yaml_file))
    assert config.judges.timeout_seconds is None
