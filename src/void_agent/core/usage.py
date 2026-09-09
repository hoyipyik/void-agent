"""What one model round-trip cost, as a value.

`input` is the whole prompt the step sent — every token of context, cached
or not — so the latest step's `input` is the context size, and a run's
total input is the bill. `cache_read` and `cache_write` say how much of
that input the provider served from, or wrote to, its cache; both are
counted inside `input`, never beside it. `output` is what the model wrote,
thinking included where the provider counts it. A provider that reports
nothing leaves the step's usage None; nothing here is estimated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Usage:
    input: int
    output: int
    cache_read: int = 0
    cache_write: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
        )


NO_USAGE = Usage(input=0, output=0)
