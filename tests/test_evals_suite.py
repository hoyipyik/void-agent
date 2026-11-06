"""A suite as its files say it: the agent, the model, the servers, the
personas a case runs as, and the cases themselves."""

from __future__ import annotations

from pathlib import Path

import pytest

from void_agent.evals.suite import SuiteError, expand, load_suite, select

CONFIG = """\
agent = "agents.orders:build"
paths = ["agents"]
cases = ["cases/manager.toml", "cases/clerk.toml"]

[model]
provider = "openai"
model = "gpt-6-luna"
effort = "max"

[context]
tool_output_limit = 2000

[servers.shop]
url = "${SHOP_MCP_URL:-http://localhost:8080/mcp}"
timeout = 60

[personas.manager]
description = "the owner"
servers.shop.headers = { Authorization = "Bearer ${TOKEN_MANAGER}" }

[personas.clerk]
exclude = ["sales_*"]
servers.shop.headers = { Authorization = "Bearer ${TOKEN_OPS}" }
"""

MANAGER = """\
persona = "manager"
tags = ["manager"]

[[cases]]
id = "manager-top-customers"
ask = "Who bought the most this year?"
tags = ["ranking"]
intent = "Ranks customers by invoiced sales, one currency at a time."
expect = { outcome = "answered", uses = ["sales_total"], avoids = ["order_list"] }

[[cases]]
id = "manager-follow-up"
turns = ["How much did Northwind buy?", { ask = "And this order?", attach = ["order.pdf"] }]
replies = ["The German one", true]
"""

CLERK = """\
[[cases]]
id = "clerk-status"
persona = "clerk"
ask = "Where is ORD-2026-0042?"
gap = "no tool says where a shipment is"
expect = { max_calls = 0 }
"""


def write(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root / "eval.toml"


def suite_in(root: Path, **overrides: str) -> Path:
    files = {"eval.toml": CONFIG, "cases/manager.toml": MANAGER, "cases/clerk.toml": CLERK}
    return write(root, {**files, **overrides})


def test_a_suite_reads_its_cases_from_the_files_it_names(tmp_path: Path) -> None:
    suite = load_suite(suite_in(tmp_path))

    assert [case.id for case in suite.cases] == [
        "manager-top-customers",
        "manager-follow-up",
        "clerk-status",
    ]
    assert suite.agent == "agents.orders:build"
    assert suite.paths == (tmp_path / "agents",)
    assert (suite.model.provider, suite.model.model, suite.model.effort) == (
        "openai",
        "gpt-6-luna",
        "max",
    )
    assert suite.context.tool_output_limit == 2000
    assert suite.runs == tmp_path / "runs"


def test_a_cases_file_gives_its_cases_a_persona_and_tags(tmp_path: Path) -> None:
    top, follow_up, status = load_suite(suite_in(tmp_path)).cases

    assert (top.persona, top.tags) == ("manager", ("manager", "ranking"))
    assert (follow_up.persona, follow_up.tags) == ("manager", ("manager",))
    assert (status.persona, status.tags) == ("clerk", ())


def test_a_question_is_a_conversation_of_one_turn(tmp_path: Path) -> None:
    top, follow_up, _ = load_suite(suite_in(tmp_path)).cases

    assert [turn.text for turn in top.turns] == ["Who bought the most this year?"]
    assert [turn.text for turn in follow_up.turns] == [
        "How much did Northwind buy?",
        "And this order?",
    ]
    assert follow_up.turns[1].attach == (tmp_path / "cases" / "order.pdf",)
    assert follow_up.replies == ("The German one", True)


def test_what_a_case_expects_is_read_with_it(tmp_path: Path) -> None:
    top, _, status = load_suite(suite_in(tmp_path)).cases

    assert top.expect.outcome == "answered"
    assert top.expect.uses == ("sales_total",)
    assert top.expect.avoids == ("order_list",)
    assert top.intent.startswith("Ranks customers")
    assert status.gap == "no tool says where a shipment is"
    assert status.expect.max_calls == 0
    assert top.expect.max_calls is None


def test_the_suite_may_name_tools_whose_calls_no_case_counts(tmp_path: Path) -> None:
    config = CONFIG + '\n[expect]\nuncounted = ["current_user"]\n'
    top, _, status = load_suite(suite_in(tmp_path, **{"eval.toml": config})).cases

    assert top.expect.uncounted == ("current_user",)
    assert status.expect.uncounted == ("current_user",)


def test_a_persona_takes_the_suites_servers_with_its_own_headers(tmp_path: Path) -> None:
    suite = load_suite(suite_in(tmp_path))

    manager = suite.personas["manager"].servers["shop"]
    assert manager.url == "${SHOP_MCP_URL:-http://localhost:8080/mcp}"
    assert manager.headers == {"Authorization": "Bearer ${TOKEN_MANAGER}"}
    assert manager.timeout == 60
    assert suite.personas["clerk"].exclude == ("sales_*",)


def test_a_case_that_runs_as_a_persona_the_suite_lacks_is_refused(tmp_path: Path) -> None:
    path = suite_in(tmp_path, **{"cases/clerk.toml": CLERK.replace('"clerk"', '"sales"')})

    with pytest.raises(SuiteError, match=r"clerk-status.*sales"):
        load_suite(path)


def test_two_cases_with_one_id_are_refused(tmp_path: Path) -> None:
    path = suite_in(
        tmp_path, **{"cases/clerk.toml": CLERK.replace("clerk-status", "manager-follow-up")}
    )

    with pytest.raises(SuiteError, match="manager-follow-up"):
        load_suite(path)


def test_a_case_with_nothing_to_ask_is_refused(tmp_path: Path) -> None:
    path = suite_in(tmp_path, **{"cases/clerk.toml": CLERK.replace("ask =", "note =")})

    with pytest.raises(SuiteError, match="clerk-status"):
        load_suite(path)


def test_a_variable_is_read_from_the_environment_or_its_default() -> None:
    env = {"TOKEN_MANAGER": "token-manager"}

    assert expand("Bearer ${TOKEN_MANAGER}", env, "manager") == "Bearer token-manager"
    assert expand("${URL:-http://localhost:8080/mcp}", env, "manager") == (
        "http://localhost:8080/mcp"
    )


def test_an_unset_variable_names_itself_and_the_persona() -> None:
    with pytest.raises(SuiteError, match=r"TOKEN_SALES.*sales"):
        expand("Bearer ${TOKEN_SALES}", {}, "sales")


def test_cases_are_chosen_by_id_persona_and_tag(tmp_path: Path) -> None:
    cases = load_suite(suite_in(tmp_path)).cases

    def ids(**by: tuple[str, ...]) -> list[str]:
        return [case.id for case in select(cases, **by)]

    assert ids(ids=("manager-*",)) == ["manager-top-customers", "manager-follow-up"]
    assert ids(personas=("clerk",)) == ["clerk-status"]
    assert ids(tags=("ranking",)) == ["manager-top-customers"]
    assert ids(ids=("manager-*",), tags=("ranking",)) == ["manager-top-customers"]
    assert ids() == ["manager-top-customers", "manager-follow-up", "clerk-status"]


def test_an_agent_named_by_its_file_is_found_beside_the_config(tmp_path: Path) -> None:
    path = suite_in(
        tmp_path, **{"eval.toml": CONFIG.replace("agents.orders:build", "agent.py:build")}
    )

    assert load_suite(path).agent == f"{tmp_path / 'agent.py'}:build"


def test_a_server_may_name_a_path_beside_the_suite(tmp_path: Path) -> None:
    stdio = CONFIG.replace(
        'url = "${SHOP_MCP_URL:-http://localhost:8080/mcp}"',
        'command = "uv"\nargs = ["run", "serve", "${SUITE_DIR}/data"]',
    )

    suite = load_suite(suite_in(tmp_path, **{"eval.toml": stdio}))

    assert suite.personas["manager"].servers["shop"].args == ("run", "serve", f"{tmp_path}/data")


def test_a_suite_names_the_model_that_plays_the_person_and_a_case_says_who_they_are(
    tmp_path: Path,
) -> None:
    config = (
        CONFIG + '\n[person]\nprovider = "anthropic"\nmodel = "claude-haiku-5-5"\nrounds = 2\n'
    )
    cases = CLERK.replace(
        'gap = "no tool', 'brief = "You mean the Contoso order."\ngap = "no tool'
    )

    suite = load_suite(suite_in(tmp_path, **{"eval.toml": config, "cases/clerk.toml": cases}))

    assert suite.person is not None
    assert (suite.person.model.provider, suite.person.model.model) == (
        "anthropic",
        "claude-haiku-5-5",
    )
    assert suite.person.rounds == 2
    assert suite.cases[2].brief == "You mean the Contoso order."
    assert load_suite(suite_in(tmp_path / "plain")).person is None


def test_a_suite_may_say_who_the_person_is_and_otherwise_names_no_trade(tmp_path: Path) -> None:
    person = '\n[person]\nprovider = "anthropic"\nmodel = "claude-haiku-5-5"\n'

    given = load_suite(
        suite_in(tmp_path / "given", **{"eval.toml": CONFIG + person + 'role = "a night nurse"\n'})
    )
    plain = load_suite(suite_in(tmp_path / "plain", **{"eval.toml": CONFIG + person}))

    assert given.person is not None and given.person.role == "a night nurse"
    assert plain.person is not None and plain.person.role == "a person who uses a chat assistant"
