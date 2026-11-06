"""A case's attachment: a file beside the cases file, sent with its turn
as the `file` part a client sends — a media type, the bytes as a base64
data URL, the filename. Images and PDFs go as they are; a text file goes
as text. The limits are the providers' own, roughly; text is capped well
below a context window."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
PDF_TYPE = "application/pdf"
TEXT_APPLICATION_TYPES = frozenset(
    {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
        "application/javascript",
    }
)
IMAGE_LIMIT = 10 * 1024 * 1024
PDF_LIMIT = 32 * 1024 * 1024
TEXT_LIMIT = 200 * 1024


def file_part(path: Path) -> dict[str, Any] | str:
    """The file as a `file` part — or, as a string, why it is not one."""
    if not path.is_file():
        return f"no such file: {path}"
    data = path.read_bytes()
    if not data:
        return f"empty file: {path.name}"
    media_type = _media_type(path, data)
    if media_type is None:
        return f"unsupported file: {path.name} — images, PDFs and text files attach"
    if media_type in IMAGE_TYPES.values():
        limit = IMAGE_LIMIT
    elif media_type == PDF_TYPE:
        limit = PDF_LIMIT
    else:
        limit = TEXT_LIMIT
    if len(data) > limit:
        return f"too large: {path.name} is {len(data)} bytes; the limit is {limit}"
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "file",
        "mediaType": media_type,
        "filename": path.name,
        "url": f"data:{media_type};base64,{encoded}",
    }


def _media_type(path: Path, data: bytes) -> str | None:
    """By extension for images and PDFs; by extension, then by content, for
    text. None for anything else."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_TYPES:
        return IMAGE_TYPES[suffix]
    if suffix == ".pdf":
        return PDF_TYPE
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed is not None and (guessed.startswith("text/") or guessed in TEXT_APPLICATION_TYPES):
        return guessed
    if guessed is None and _is_text(data):
        return "text/plain"
    return None


def _is_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True
