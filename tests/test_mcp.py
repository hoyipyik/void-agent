"""The MCP bridge: a server's tools become `Tool`s, and the approval that
guards them is declared here, in our code — never by the server."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent
from tests.mcp_fakes import WRITE_SCHEMA, FakeMcp, descriptor, text_result

from void_agent import (
    Call,
    EventSender,
    Internal,
    Rejected,
    ScriptedHuman,
    attended,
    public_text,
)
from void_agent.mcp import McpServer, McpUnknownTool


def must_sign(input: Any) -> str | None:
    return f"writing {input['path']} needs a signature"


async def test_a_discovered_tool_carries_the_servers_name_description_and_schema() -> None:
    fake = FakeMcp([descriptor("write_file")])
    async with McpServer(client=fake) as server:
        capability = server.tool("write_file")
    assert capability.name == "write_file"
    assert capability.description == "writes a file"
    assert capability.input_schema == WRITE_SCHEMA


async def test_the_tools_are_discovered_once_when_the_server_is_mounted() -> None:
    fake = FakeMcp([descriptor("write_file"), descriptor("read_file", "reads a file")])
    async with McpServer(client=fake) as server:
        assert server.names == ("write_file", "read_file")
        assert len(server.tools()) == 2
        server.tool("write_file")
    assert fake.listed == 1


async def test_the_session_is_opened_and_closed_around_the_block() -> None:
    fake = FakeMcp([descriptor("write_file")])
    async with McpServer(client=fake):
        assert fake.open and not fake.closed
    assert fake.closed


async def test_calling_the_tool_sends_the_validated_arguments_to_the_server() -> None:
    fake = FakeMcp([descriptor("write_file")])
    async with McpServer(client=fake) as server:
        assert await server.tool("write_file").invoke({"path": "/a", "text": "hi"}, EventSender())
    assert fake.calls == [("write_file", {"path": "/a", "text": "hi"})]


async def test_a_single_text_block_comes_back_as_the_tools_result() -> None:
    fake = FakeMcp([descriptor("write_file")], text_result("wrote 3 bytes"))
    async with McpServer(client=fake) as server:
        assert await server.tool("write_file").invoke({"path": "/a"}, EventSender()) == (
            "wrote 3 bytes"
        )


async def test_structured_content_is_the_result_when_the_server_returns_it() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="wrote 3 bytes")],
        structured_content={"bytes": 3},
    )
    fake = FakeMcp([descriptor("write_file")], result)
    async with McpServer(client=fake) as server:
        assert await server.tool("write_file").invoke({"path": "/a"}, EventSender()) == {
            "bytes": 3
        }


async def test_a_binary_block_is_described_not_inlined_into_the_transcript() -> None:
    result = CallToolResult(
        content=[ImageContent(type="image", data="qqqq", mime_type="image/png")], is_error=False
    )
    fake = FakeMcp([descriptor("screenshot")], result)
    async with McpServer(client=fake) as server:
        value = await server.tool("screenshot").invoke({"path": "/a"}, EventSender())
    assert value == [{"type": "image", "media_type": "image/png", "bytes": 3}]


async def test_a_server_error_reaches_the_model_as_a_rejection() -> None:
    fake = FakeMcp([descriptor("write_file")], text_result("no such directory", is_error=True))
    async with McpServer(client=fake) as server:
        with pytest.raises(Rejected, match="write_file: no such directory"):
            await server.tool("write_file").invoke({"path": "/a"}, EventSender())


async def test_a_transport_failure_stays_internal() -> None:
    fake = FakeMcp([descriptor("write_file")], failure=RuntimeError("broken pipe"))
    async with McpServer(client=fake) as server:
        with pytest.raises(Internal) as internal:
            await server.tool("write_file").invoke({"path": "/a"}, EventSender())
    assert "broken pipe" not in public_text(internal.value)


async def test_the_approval_is_declared_here_and_a_declined_call_never_reaches_the_server() -> (
    None
):
    fake = FakeMcp([descriptor("write_file")])
    async with McpServer(client=fake) as server:
        capability = server.tool("write_file", approval=must_sign)
        with (
            attended(ScriptedHuman([False])),
            pytest.raises(Rejected, match="declined write_file"),
        ):
            await capability.invoke({"path": "/etc/hosts"}, EventSender())
    assert fake.calls == []


async def test_a_signed_call_carries_the_exact_arguments_on_its_card() -> None:
    fake = FakeMcp([descriptor("write_file")])
    human = ScriptedHuman([True])
    async with McpServer(client=fake) as server:
        capability = server.tool("write_file", approval=must_sign)
        with attended(human):
            await capability.invoke({"path": "/etc/hosts"}, EventSender())
    [ask] = human.asked
    assert ask.kind == "approval"
    assert ask.question == "writing /etc/hosts needs a signature"
    assert ask.call == Call(tool="write_file", input={"path": "/etc/hosts"})
    assert fake.calls == [("write_file", {"path": "/etc/hosts"})]


async def test_only_the_named_tools_are_gated() -> None:
    fake = FakeMcp([descriptor("write_file"), descriptor("read_file", "reads a file")])
    async with McpServer(client=fake) as server:
        write, read = server.tools(approvals={"write_file": must_sign})
        with attended(ScriptedHuman([])):
            assert await read.invoke({"path": "/a"}, EventSender())
    assert [name for name, _ in fake.calls] == ["read_file"]
    assert write.name == "write_file"


async def test_an_approval_for_a_tool_the_server_lacks_is_an_error_at_mount() -> None:
    fake = FakeMcp([descriptor("read_file", "reads a file")])
    async with McpServer(client=fake) as server:
        with pytest.raises(McpUnknownTool, match="read_file"):
            server.tools(approvals={"write_file": must_sign})
        with pytest.raises(McpUnknownTool, match="read_file"):
            server.tool("write_file")


async def test_a_tool_without_a_description_falls_back_to_its_name() -> None:
    fake = FakeMcp([descriptor("write_file", None)])
    async with McpServer(client=fake) as server:
        assert server.tool("write_file").description == "write_file"


async def test_the_tools_are_only_available_while_the_server_is_mounted() -> None:
    server = McpServer(client=FakeMcp([descriptor("write_file")]))
    with pytest.raises(RuntimeError, match="not mounted"):
        server.tool("write_file")


async def test_a_server_can_be_namespaced_so_two_of_them_never_collide() -> None:
    fake = FakeMcp([descriptor("write_file"), descriptor("read_file", "reads a file")])
    async with McpServer(client=fake) as server:
        write, read = server.tools(prefix="files__")
        assert (write.name, read.name) == ("files__write_file", "files__read_file")
        await write.invoke({"path": "/a"}, EventSender())
    # The server is called by the name it knows itself, never by the alias.
    assert fake.calls == [("write_file", {"path": "/a"})]


async def test_an_alias_renames_one_tool_for_the_model() -> None:
    fake = FakeMcp([descriptor("write_file")])
    async with McpServer(client=fake) as server:
        capability = server.tool("write_file", alias="files__write_file")
        await capability.invoke({"path": "/a"}, EventSender())
    assert capability.name == "files__write_file"
    assert fake.calls == [("write_file", {"path": "/a"})]


async def test_the_scalar_wrapper_a_server_puts_round_a_plain_value_is_unwrapped() -> None:
    """MCP wraps a non-object return in `{"result": …}`; the model wants
    the value, not the envelope."""
    result = CallToolResult(
        content=[TextContent(type="text", text="wrote 3 bytes")],
        structured_content={"result": "wrote 3 bytes"},
    )
    fake = FakeMcp([descriptor("write_file")], result)
    async with McpServer(client=fake) as server:
        assert await server.tool("write_file").invoke({"path": "/a"}, EventSender()) == (
            "wrote 3 bytes"
        )


async def test_a_structured_object_of_its_own_is_kept_whole() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="{}")],
        structured_content={"result": 1, "bytes": 3},
    )
    fake = FakeMcp([descriptor("write_file")], result)
    async with McpServer(client=fake) as server:
        value = await server.tool("write_file").invoke({"path": "/a"}, EventSender())
    assert value == {"result": 1, "bytes": 3}


async def test_a_servers_own_noise_can_be_sent_somewhere_other_than_the_terminal(
    tmp_path: Path,
) -> None:
    """A stdio server's stderr goes to the parent's by default, which in a
    terminal UI is the canvas being drawn on. It must be divertible."""
    log = tmp_path / "noisy.log"
    server = McpServer.stdio(
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('server said something\\n'); sys.exit(3)",
        errlog=log,
    )
    with pytest.raises(Exception):  # noqa: B017 - it never speaks the protocol
        async with server:
            pass
    assert "server said something" in log.read_text(encoding="utf-8")
