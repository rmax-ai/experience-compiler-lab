"""Tests for the M5 ablation configuration matrix."""

from experiments.configs import CONFIG_MATRIX, AblationConfig, config_help


def test_config_matrix_matches_specification() -> None:
    assert CONFIG_MATRIX == {
        "baseline": {
            "persist_traces": False,
            "mine_knowledge": False,
            "evolve_skills": False,
            "knowledge_in_execution": False,
        },
        "trace2skill": {
            "persist_traces": True,
            "mine_knowledge": False,
            "evolve_skills": True,
            "knowledge_in_execution": False,
        },
        "memory": {
            "persist_traces": True,
            "mine_knowledge": True,
            "evolve_skills": False,
            "knowledge_in_execution": True,
        },
        "compiler": {
            "persist_traces": True,
            "mine_knowledge": True,
            "evolve_skills": True,
            "knowledge_in_execution": False,
        },
    }


def test_all_ablation_configs_map_to_matrix_rows() -> None:
    assert {config.value for config in AblationConfig} == set(CONFIG_MATRIX)


def test_config_help_describes_each_configuration() -> None:
    help_text = config_help()
    assert help_text
    for config in AblationConfig:
        assert config.value in help_text
