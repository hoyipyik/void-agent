"""The OS clipboard, read on demand — ctrl+v or `/paste`.

A terminal paste never carries an image: what arrives is text. So an
image (a screenshot, a copied picture) or a file copied in the file
manager is fetched from the OS clipboard instead — a copied file first,
else an image — each as an `Attachment`. macOS goes through `osascript`,
Linux through `wl-paste` then `xclip`, Windows through PowerShell. The
runner is a seam: tests hand in a fake.
"""

from __future__ import annotations

import base64
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlparse

from cli.attachments import Attachment, read_attachment

Runner = Callable[[list[str]], bytes | None]

NOTHING = "nothing to paste: no file or image on the clipboard"

_PNG_HEX = re.compile(rb"PNGf([0-9A-Fa-f]+)")

_MAC_FILE = "POSIX path of (the clipboard as «class furl»)"
_MAC_PNG = "the clipboard as «class PNGf»"
_WIN_FILES = "Get-Clipboard -Format FileDropList | ForEach-Object { $_.FullName }"
_WIN_PNG = (
    "$image = Get-Clipboard -Format Image; if ($image) {"
    " $stream = New-Object System.IO.MemoryStream;"
    " $image.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png);"
    " [Convert]::ToBase64String($stream.ToArray()) }"
)


def run_command(command: list[str]) -> bytes | None:
    """stdout on success; None when the tool is missing, fails, or has
    nothing of that kind to give."""
    try:
        completed = subprocess.run(command, capture_output=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _existing(candidates: list[str]) -> list[Path]:
    paths = [Path(raw.strip()) for raw in candidates if raw.strip()]
    return [path for path in paths if path.is_file()]


def _from_uri_list(raw: bytes) -> list[str]:
    paths: list[str] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("file://"):
            paths.append(unquote(urlparse(line).path))
    return paths


class Clipboard:
    def __init__(self, runner: Runner | None = None, platform: str | None = None) -> None:
        self._run: Runner = runner or run_command
        self._platform = platform or sys.platform

    def read(self) -> list[Attachment] | str:
        """The copied files, else the copied image, else why there is
        nothing to attach."""
        paths = self.files()
        if paths:
            attachments: list[Attachment] = []
            for path in paths:
                read = read_attachment(path)
                if isinstance(read, str):
                    return read
                attachments.append(read)
            return attachments
        image = self.image()
        return [image] if image is not None else NOTHING

    def files(self) -> list[Path]:
        if self._platform == "darwin":
            out = self._run(["osascript", "-e", _MAC_FILE])
            return _existing([out.decode("utf-8", errors="replace")]) if out else []
        if self._platform.startswith("linux"):
            out = self._run(["wl-paste", "--type", "text/uri-list"]) or self._run(
                ["xclip", "-selection", "clipboard", "-t", "text/uri-list", "-o"]
            )
            return _existing(_from_uri_list(out)) if out else []
        if self._platform == "win32":
            out = self._run(["powershell", "-NoProfile", "-Command", _WIN_FILES])
            return _existing(out.decode("utf-8", errors="replace").splitlines()) if out else []
        return []

    def image(self) -> Attachment | None:
        data: bytes | None = None
        if self._platform == "darwin":
            out = self._run(["osascript", "-e", _MAC_PNG]) or b""
            found = _PNG_HEX.search(out)
            if found is not None:
                try:
                    data = bytes.fromhex(found.group(1).decode("ascii"))
                except ValueError:
                    data = None
        elif self._platform.startswith("linux"):
            data = self._run(["wl-paste", "--type", "image/png"]) or self._run(
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]
            )
        elif self._platform == "win32":
            out = self._run(["powershell", "-NoProfile", "-Command", _WIN_PNG])
            if out and out.strip():
                try:
                    data = base64.b64decode(out.strip(), validate=True)
                except ValueError:
                    data = None
        if not data:
            return None
        return Attachment("clipboard.png", "image/png", data)
