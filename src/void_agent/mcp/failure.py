"""A mount that failed, as a type of ours.

The SDK opens its transports inside task groups, so a mount that fails
comes out as an exception group, nested, holding the SDK's and the HTTP
client's own types — and with the one fact a caller takes a different road
on already gone: the SDK folds a 401 into the same error as a 500. A
refused token is asked for again; a server that is not there is simply
down. So `server.py` notes each status where the response arrives, and
the failure is filed here, with what was raised kept as its cause.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

import httpx2


class McpMountFailed(Exception):
    """The server could not be mounted. `status` is the HTTP status it
    refused the mount with, when it answered at all. What the SDK raised
    is the `__cause__`: detail for a log, never for a decision."""

    def __init__(self, reason: str, *, status: int | None = None) -> None:
        super().__init__(reason)
        self.status = status


class McpUnauthorized(McpMountFailed):
    """The server refused the credentials it was shown (HTTP 401): a token
    that expired or was revoked, or none at all. A 403 is not this — the
    server knew who asked and said no, and asking again changes nothing."""


class McpNotFound(McpMountFailed):
    """Nothing serves MCP at that address (HTTP 404)."""


class McpUnreachable(McpMountFailed):
    """Nothing answered: the connection was refused, the host is unknown,
    or the process would not start."""


# What a transport raises when there was no answer to read a status from.
NO_ANSWER = (httpx2.TransportError, OSError)


def mount_failure(raised: Exception, answers: Sequence[int]) -> McpMountFailed:
    """What stopped the mount, filed. `answers` are the statuses the server
    gave the mount's requests, in order: the last one is what ended it."""
    leaves = tuple(_leaves(raised))
    said = "; ".join(str(leaf) or type(leaf).__name__ for leaf in leaves)
    status = answers[-1] if answers and answers[-1] >= 400 else None
    if status == 401:
        return McpUnauthorized(
            "the server refused the credentials it was shown (HTTP 401)", status=status
        )
    if status == 404:
        return McpNotFound("nothing serves MCP at that address (HTTP 404)", status=status)
    if status is not None:
        return McpMountFailed(f"{said} (HTTP {status})", status=status)
    if any(isinstance(leaf, NO_ANSWER) for leaf in leaves):
        return McpUnreachable(said)
    return McpMountFailed(said)


def _leaves(raised: BaseException) -> Iterator[BaseException]:
    """A transport's failure is a group that only says how many there
    were: its leaves are the words."""
    if isinstance(raised, BaseExceptionGroup):
        for inner in cast("BaseExceptionGroup[BaseException]", raised).exceptions:
            yield from _leaves(inner)
    else:
        yield raised
