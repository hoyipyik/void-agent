"""Where the CLI's provider comes from: the environment first, then the
config file the key prompt and `/model` write."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from cli.config import Config, load_config, save_config
from cli.llm import resolve_llm


def test_the_environment_names_the_provider(tmp_path: Path) -> None:
    config = load_config({"ANTHROPIC_API_KEY": "sk-env"}, tmp_path / "config.json")
    assert config.provider == "anthropic"
    assert config.api_key == "sk-env"
    assert config.model == "claude-opus-5"


def test_an_openai_key_alone_names_openai_with_its_model_and_base_url(tmp_path: Path) -> None:
    config = load_config(
        {"OPENAI_API_KEY": "k", "OPENAI_MODEL": "m", "OPENAI_BASE_URL": "http://x"},
        tmp_path / "config.json",
    )
    assert (config.provider, config.model, config.openai_base_url) == ("openai", "m", "http://x")


def test_the_openai_reasoning_effort_comes_from_the_environment_or_the_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    save_config(path, Config(provider="openai", openai_api_key="k", openai_reasoning_effort="low"))
    assert load_config({}, path).openai_reasoning_effort == "low"
    assert load_config({"OPENAI_REASONING_EFFORT": "none"}, path).openai_reasoning_effort == "none"


def test_the_agent_is_the_files_choice_or_the_default(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    assert load_config({}, path).agent == "universal"
    save_config(path, Config(agent="deep"))
    assert load_config({}, path).agent == "deep"  # VOID_AGENT is the entry point's to read


def test_the_context_tool_output_limit_is_the_files_or_the_default(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    assert load_config({}, path).tool_output_limit() == 8_000
    save_config(path, Config(context_tool_output_limit=20_000))
    assert load_config({}, path).tool_output_limit() == 20_000
    save_config(path, Config(context_tool_output_limit=0))
    assert load_config({}, path).tool_output_limit() is None  # 0 lifts the cap
    path.write_text('{"context_tool_output_limit": "lots"}', encoding="utf-8")
    assert load_config({}, path).context_tool_output_limit == 8_000


def test_a_saved_config_fills_in_what_the_environment_lacks(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    save_config(path, Config(provider="openai", openai_api_key="k", openai_model="m"))
    config = load_config({}, path)
    assert (config.provider, config.api_key, config.model) == ("openai", "k", "m")


def test_the_environment_wins_over_the_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    save_config(path, Config(provider="anthropic", anthropic_api_key="from-file"))
    config = load_config({"ANTHROPIC_API_KEY": "from-env"}, path)
    assert config.api_key == "from-env"


def test_no_key_anywhere_means_no_provider(tmp_path: Path) -> None:
    config = load_config({}, tmp_path / "missing.json")
    assert config.provider is None
    assert config.api_key == ""
    assert resolve_llm(config) is None


def test_the_config_file_is_private(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    save_config(path, Config(provider="anthropic", anthropic_api_key="secret"))
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_switching_the_model_keeps_the_other_providers_settings() -> None:
    config = Config(provider="anthropic", anthropic_api_key="a", openai_api_key="o")
    switched = config.with_model("openai", "gpt-4o-mini")
    assert (switched.provider, switched.model, switched.api_key) == ("openai", "gpt-4o-mini", "o")
    assert switched.anthropic_api_key == "a"


def test_an_ollama_model_in_the_environment_names_ollama_when_no_key_is_set(
    tmp_path: Path,
) -> None:
    config = load_config(
        {"OLLAMA_MODEL": "qwen3:8b", "OLLAMA_HOST": "box:11434"}, tmp_path / "config.json"
    )
    assert (config.provider, config.model, config.ollama_host) == (
        "ollama",
        "qwen3:8b",
        "http://box:11434",
    )
    assert config.configured()
    assert config.api_key == ""  # a local server: nothing to hold
    assert resolve_llm(config) is not None


def test_a_key_outranks_ollama_unless_the_file_chose_it(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    env = {"ANTHROPIC_API_KEY": "k", "OLLAMA_MODEL": "qwen3:8b"}
    assert load_config(env, path).provider == "anthropic"
    save_config(path, Config(provider="ollama", ollama_model="qwen3:8b"))
    assert load_config(env, path).provider == "ollama"


def test_ollama_without_a_chosen_model_is_not_a_provider_yet(tmp_path: Path) -> None:
    config = load_config({"OLLAMA_HOST": "localhost"}, tmp_path / "config.json")
    assert config.provider is None
    assert config.ollama_host == "http://localhost:11434"


def test_switching_to_an_ollama_model_needs_no_key() -> None:
    config = Config(provider="anthropic", anthropic_api_key="a").with_model("ollama", "qwen3:8b")
    assert (config.provider, config.model, config.configured()) == ("ollama", "qwen3:8b", True)
    assert config.anthropic_api_key == "a"


def test_a_key_is_kept_under_its_provider() -> None:
    config = Config().with_key("openai", "k")
    assert (config.provider, config.openai_api_key, config.anthropic_api_key) == (
        "openai",
        "k",
        "",
    )


# ── what the person turned on and off in /mcp and /skill ─────────────────


def test_a_server_is_on_until_it_is_turned_off() -> None:
    config = Config()
    assert config.server_state("files") == "on"
    off = config.with_server_state("files", "off")
    assert off.server_state("files") == "off"
    assert off.mcp_off_servers == ("files",)
    assert off.with_server_state("files", "on").mcp_off_servers == ()


def test_a_skill_is_on_until_it_is_turned_off() -> None:
    config = Config()
    assert config.skill_state("refunds") == "on"
    off = config.with_skill_state("refunds", "off")
    assert off.skill_state("refunds") == "off"
    assert off.skills_off == ("refunds",)
    assert off.enabled_skills(("refunds", "pi-drafting")) == frozenset({"pi-drafting"})


def test_the_servers_and_skills_survive_the_config_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    save_config(
        path,
        Config().with_server_state("docs", "off").with_skill_state("refunds", "off"),
    )
    loaded = load_config({}, path)
    assert loaded.server_state("docs") == "off"
    assert loaded.skill_state("refunds") == "off"


# ── the MCP tools the person turned on, off, or marked as signed ─────────


def test_an_unmarked_tool_takes_the_servers_default() -> None:
    """A tool nobody has judged is not simply on: it falls to whatever the
    server it came from was given, and a server given nothing is signed."""
    config = Config()
    assert config.tool_state("files__write_file", "signed") == "signed"
    assert config.tool_state("notion__search", "on") == "on"
    assert config.tool_state("files__write_file") == "signed"  # the fallback's own default


def test_a_mark_of_the_persons_own_outranks_the_default() -> None:
    config = Config().with_tool_state("files__read_file", "on")
    assert config.tool_state("files__read_file", "signed") == "on"
    assert config.mcp_on == ("files__read_file",)
    assert config.tool_state("files__write_file", "signed") == "signed"

    off = config.with_tool_state("files__read_file", "off")
    assert off.tool_state("files__read_file", "signed") == "off"
    assert off.mcp_on == ()  # the three marks are exclusive


def test_marking_a_tool_signed_replaces_whatever_state_it_had() -> None:
    config = Config().with_tool_state("files__write_file", "off")
    signed = config.with_tool_state("files__write_file", "signed")
    assert signed.tool_state("files__write_file") == "signed"
    assert signed.mcp_off == ()
    assert signed.mcp_signed == ("files__write_file",)
    on = signed.with_tool_state("files__write_file", "on")
    assert on.mcp_signed == () and on.mcp_on == ("files__write_file",)


def test_the_states_survive_a_round_trip_through_the_config_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = (
        Config(provider="anthropic", anthropic_api_key="k")
        .with_tool_state("files__write_file", "signed")
        .with_tool_state("docs__delete", "off")
    )
    save_config(path, config)
    loaded = load_config({}, path)
    assert loaded.tool_state("files__write_file") == "signed"
    assert loaded.tool_state("docs__delete") == "off"


def test_a_config_written_before_marks_were_three_still_reads() -> None:
    """mcp_on is new; a file that predates it has no explicit allowances."""
    assert Config.from_json({"mcp_signed": ["a"], "mcp_off": ["b"]}).mcp_on == ()
