"""Provider-neutral message content. Binary data is encoded only at the API boundary.

Callers read files explicitly; content never opens paths or fetches URLs.
Model and endpoint support, file validity and request limits are enforced by
providers. These values describe inputs, not generated output parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


def _validate_data(data: object) -> None:
    if not isinstance(data, bytes) or not data:
        raise ValueError("content data must be nonempty bytes")


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text content must not be empty")


@dataclass(frozen=True, slots=True)
class ImageContent:
    """An image's original bytes and MIME type, not a path or base64 string."""

    data: bytes = field(repr=False)
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]

    def __post_init__(self) -> None:
        _validate_data(self.data)
        if self.media_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            raise ValueError(f"unsupported image media type: {self.media_type}")


@dataclass(frozen=True, slots=True)
class PdfContent:
    """A PDF sent intact so the model can inspect both text and page visuals."""

    data: bytes = field(repr=False)
    filename: str = "document.pdf"

    def __post_init__(self) -> None:
        _validate_data(self.data)
        if not self.filename.strip():
            raise ValueError("PDF filename must not be empty")


ContentPart = TextContent | ImageContent | PdfContent
