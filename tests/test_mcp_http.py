"""The bridge over a real HTTP transport: the token a mount carries, what a
server sent over the wire, what a mount that fails is, and how long a
server may take. Whoever mounts a server takes a different road for each
failure — a refused token is asked for again, a server that is not there
is simply down — so each is a type of ours, never the SDK's exception
group."""

from __future__ import annotations

import asyncio

import pytest
from starlette.types import Receive, Scope, Send
from tests.mcp_http import (
    BEARER,
    TOKEN,
    VECTORS,
    Published,
    answering,
    nobody_listening,
    serving,
    silent,
)

from void_agent import EventSender, Internal, public_text
from void_agent.mcp import (
    McpMountFailed,
    McpNotFound,
    McpServer,
    McpUnauthorized,
    McpUnreachable,
)


async def test_a_hosted_server_is_mounted_with_the_headers_it_wants() -> None:
    published = Published()
    async with (
        serving(published) as url,
        McpServer.http(f"{url}/mcp", headers=BEARER) as server,
    ):
        assert server.names == tuple(example["tool"]["name"] for example in VECTORS["tools"])
        assert server.instructions == VECTORS["instructions"]["text"]
        assert await server.tool("ping").invoke({}, EventSender()) == "pong"
    assert set(published.shown) == {f"Bearer {TOKEN}"}


async def test_the_fingerprints_are_of_what_the_server_sent_over_the_wire() -> None:
    async with (
        serving(Published()) as url,
        McpServer.http(f"{url}/mcp", headers=BEARER) as server,
    ):
        computed = {name: server.fingerprint(name) for name in server.names}
        about_itself = server.instructions_fingerprint
    assert computed == {
        example["tool"]["name"]: example["fingerprint"] for example in VECTORS["tools"]
    }
    assert about_itself == VECTORS["instructions"]["fingerprint"]


async def test_a_server_that_refuses_the_token_is_unauthorized() -> None:
    async with serving(Published()) as url:
        with pytest.raises(McpUnauthorized) as refused:
            async with McpServer.http(f"{url}/mcp", headers={"Authorization": "Bearer stale"}):
                pass
    assert refused.value.status == 401


async def test_a_server_shown_no_token_at_all_is_unauthorized() -> None:
    async with serving(Published()) as url:
        with pytest.raises(McpUnauthorized):
            async with McpServer.http(f"{url}/mcp"):
                pass


async def test_an_address_that_serves_no_mcp_is_not_found() -> None:
    async with serving(answering(404)) as url:
        with pytest.raises(McpNotFound) as missing:
            async with McpServer.http(f"{url}/mcp", headers=BEARER):
                pass
    assert missing.value.status == 404


async def test_a_port_nobody_listens_on_is_unreachable() -> None:
    with pytest.raises(McpUnreachable) as down:
        async with McpServer.http(f"{nobody_listening()}/mcp", headers=BEARER):
            pass
    assert down.value.status is None


async def test_any_other_refusal_is_a_failed_mount_that_carries_the_status() -> None:
    async with serving(answering(500)) as url:
        with pytest.raises(McpMountFailed) as failed:
            async with McpServer.http(f"{url}/mcp", headers=BEARER):
                pass
    assert type(failed.value) is McpMountFailed
    assert failed.value.status == 500
    assert "500" in str(failed.value)


async def test_each_kind_of_failed_mount_is_a_failed_mount() -> None:
    """One `except` for a caller that takes the same road for all of them."""
    for kind in (McpUnauthorized, McpNotFound, McpUnreachable):
        assert issubclass(kind, McpMountFailed)


async def test_a_failed_mount_keeps_what_was_raised_as_its_cause() -> None:
    """The detail is for a log: nothing of the SDK's is lost, only wrapped."""
    with pytest.raises(McpUnreachable) as down:
        async with McpServer.http(f"{nobody_listening()}/mcp", headers=BEARER):
            pass
    assert isinstance(down.value.__cause__, BaseExceptionGroup)


async def test_a_failed_mount_never_says_the_token() -> None:
    for app in (answering(401), answering(404), answering(500)):
        async with serving(app) as url:
            with pytest.raises(McpMountFailed) as failed:
                async with McpServer.http(f"{url}/mcp", headers=BEARER):
                    pass
        assert TOKEN not in str(failed.value)
        assert TOKEN not in repr(failed.value)


async def test_a_failed_mount_leaves_nothing_mounted() -> None:
    server = McpServer.http(f"{nobody_listening()}/mcp", headers=BEARER)
    with pytest.raises(McpUnreachable):
        async with server:
            pass
    with pytest.raises(RuntimeError, match="not mounted"):
        server.tool("ping")


async def test_cancelling_a_mount_is_cancellation_and_not_a_failed_mount() -> None:
    """Cancellation outranks every failure: a mount cut short by the task
    that wanted it is never reported as a server that could not be mounted."""
    arrived = asyncio.Event()

    async def waiting(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            arrived.set()
        await silent(scope, receive, send)

    async def mount(url: str) -> None:
        async with McpServer.http(f"{url}/mcp", headers=BEARER):
            pass

    async with serving(waiting) as url:
        mounting = asyncio.create_task(mount(url))
        await arrived.wait()
        mounting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await mounting


# Long enough for a server on this machine to answer, short enough that a
# test of one that never does is over at once.
SOON = 0.5


async def test_a_server_that_does_not_answer_the_mount_in_time_is_unreachable() -> None:
    async with serving(silent) as url:
        with pytest.raises(McpUnreachable) as late:
            async with McpServer.http(f"{url}/mcp", headers=BEARER, timeout=SOON):
                pass
    assert "in time" in str(late.value)
    assert late.value.status is None


async def test_a_call_the_server_does_not_answer_in_time_fails_and_stays_internal() -> None:
    """A stuck call must not hold the turn for the SDK's five minutes. Like
    any transport failure it is not the model's business: the run fails."""
    published = Published(stalls=frozenset({"numbers"}))
    async with (
        serving(published) as url,
        McpServer.http(f"{url}/mcp", headers=BEARER, timeout=SOON) as server,
    ):
        with pytest.raises(Internal) as internal:
            await server.tool("numbers").invoke({}, EventSender())
        published.release()
    assert "timed out" in str(internal.value)
    assert "timed out" not in public_text(internal.value)


async def test_a_server_that_answers_in_time_is_not_hurried() -> None:
    published = Published(stalls=frozenset({"numbers"}))
    async with (
        serving(published) as url,
        McpServer.http(f"{url}/mcp", headers=BEARER, timeout=SOON) as server,
    ):
        assert await server.tool("ping").invoke({}, EventSender()) == "pong"
        with pytest.raises(Internal):
            await server.tool("numbers").invoke({}, EventSender())
        # The call that timed out cost only itself: the next one is answered.
        assert await server.tool("ping").invoke({}, EventSender()) == "pong"
        published.release()
