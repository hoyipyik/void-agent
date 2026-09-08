"""Attachments the CLI reads for the person.

A file becomes the `file` part the protocol carries: a media type, the
bytes as a base64 data URL, the filename. Images and PDFs go as they are;
a text file goes as text. Core never opens a path — this is the edge that
does. Three ways a file arrives: a path pasted or dragged into the
composer (`paths_in`), an `@path` mention in the message (`mentions`), and
the OS clipboard (`cli/clipboard.py`).
"""

from __future__ import annotations

import base64
import mimetypes
import re
import shlex
from dataclasses import dataclass
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

# Provider limits, roughly: 5 MB an image, 32 MB a PDF; text is the context
# window's problem, so it is capped well below it.
IMAGE_LIMIT = 5 * 1024 * 1024
PDF_LIMIT = 32 * 1024 * 1024
TEXT_LIMIT = 200 * 1024


def size_text(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


@dataclass(frozen=True, slots=True)
class Attachment:
    name: str
    media_type: str
    data: bytes

    def part(self) -> dict[str, Any]:
        """The `file` part: what the session stores and the model reads."""
        encoded = base64.b64encode(self.data).decode("ascii")
        return {
            "type": "file",
            "mediaType": self.media_type,
            "filename": self.name,
            "url": f"data:{self.media_type};base64,{encoded}",
        }

    @property
    def size(self) -> str:
        return size_text(len(self.data))


def _is_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def media_type_of(path: Path, data: bytes) -> str | None:
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


def read_attachment(path: Path) -> Attachment | str:
    """The file as an attachment — or, as a string, why it is not one."""
    path = path.expanduser()
    if not path.is_file():
        return f"no such file: {path}"
    data = path.read_bytes()
    if not data:
        return f"empty file: {path.name}"
    media_type = media_type_of(path, data)
    if media_type is None:
        return f"unsupported file: {path.name} — images, PDFs and text files attach"
    if media_type in IMAGE_TYPES.values():
        limit = IMAGE_LIMIT
    elif media_type == PDF_TYPE:
        limit = PDF_LIMIT
    else:
        limit = TEXT_LIMIT
    if len(data) > limit:
        return f"too large: {path.name} is {size_text(len(data))}; the limit is {size_text(limit)}"
    return Attachment(path.name, media_type, data)


def paths_in(text: str) -> list[Path]:
    """The files a paste names — one absolute (or `~`) path per line, the
    way a drag from a file manager arrives, shell-quoted or not — or
    nothing, if any line is not one. A relative name is a word: pasting
    "Makefile" is not attaching it."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    paths: list[Path] = []
    for line in lines:
        path = _file_named(line)
        if path is None:
            return []
        paths.append(path)
    return paths


def _file_named(line: str) -> Path | None:
    """The file one pasted line names: as written (spaces and all), or
    shell-quoted / escaped as a drag from a file manager writes it."""
    candidates = [line]
    try:
        tokens = shlex.split(line)
    except ValueError:
        tokens = []
    if len(tokens) == 1:
        candidates.append(tokens[0])
    for raw in candidates:
        if not raw.startswith(("/", "~")):
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            return path
    return None


_MENTION = re.compile(r"(?<!\S)@([^\s@]+)[ \t]?")


def mentions(text: str) -> tuple[str, list[Path]]:
    """`@path` tokens that name a file, taken out of the text; any other
    `@word` stays as written."""
    found: list[Path] = []

    def take(match: re.Match[str]) -> str:
        candidate = Path(match.group(1)).expanduser()
        if candidate.is_file():
            found.append(candidate)
            return ""
        return match.group(0)

    if not found and "@" not in text:
        return text, found
    stripped = _MENTION.sub(take, text)
    if not found:
        return text, found
    return "\n".join(line.rstrip() for line in stripped.splitlines()).strip(), found
