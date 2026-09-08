"""The `file` part's anatomy — the UIMessage attachment a client sends:

    {"type": "file", "mediaType": "image/png",
     "url": "data:image/png;base64,…", "filename": "shot.png"}

Both model-facing projections read it here: `context_text` names it,
`context_content` decodes it into the content part the model receives —
an image, a PDF, or a text file as fenced text under its filename. The
URL must be a base64 data URL — a remote URL is never fetched, and the
runtime never opens a path."""

from __future__ import annotations

import base64
import binascii
from typing import Any, Literal, cast

from void_agent.core.content import ImageContent, PdfContent, TextContent

IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
PDF_MEDIA_TYPE = "application/pdf"
# Text attachments travel as text: any `text/*`, and the structured kinds
# that are text in all but name. The model reads them fenced, by filename.
TEXT_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
        "application/javascript",
    }
)

_BASE64_MARK = ";base64"


def _data_url(url: object) -> tuple[str | None, str | None]:
    """(media type, base64 payload) of a base64 data URL; (None, None) for
    anything else."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    header, separator, payload = url[len("data:") :].partition(",")
    if not separator or not header.endswith(_BASE64_MARK):
        return None, None
    media_type = header[: -len(_BASE64_MARK)]
    return (media_type or None), payload


def attachment_media_type(part: dict[str, Any]) -> str | None:
    """The declared `mediaType`, else the data URL's own."""
    declared = part.get("mediaType")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    return _data_url(part.get("url"))[0]


def attachment_label(part: dict[str, Any]) -> str:
    """`attachment: shot.png (image/png)`, or `attachment (image/png)` when
    the part carries no filename."""
    kind = attachment_media_type(part) or "unknown type"
    filename = str(part.get("filename") or "").strip()
    return f"attachment: {filename} ({kind})" if filename else f"attachment ({kind})"


def is_text_media_type(kind: str | None) -> bool:
    return kind is not None and (kind.startswith("text/") or kind in TEXT_MEDIA_TYPES)


def decode_attachment(part: dict[str, Any]) -> ImageContent | PdfContent | TextContent | str:
    """The content part an attachment carries — or, as a string, the reason
    it cannot be sent: a media type no provider takes, or data that is not
    a readable base64 data URL (a text file must be UTF-8)."""
    kind = attachment_media_type(part)
    if kind not in IMAGE_MEDIA_TYPES and kind != PDF_MEDIA_TYPE and not is_text_media_type(kind):
        return "unsupported media type"
    _, payload = _data_url(part.get("url"))
    if payload is None:
        return "unreadable data"
    try:
        data = base64.b64decode(payload, validate=True)
    except binascii.Error:
        return "unreadable data"
    if not data:
        return "unreadable data"
    filename = str(part.get("filename") or "").strip()
    if is_text_media_type(kind):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return "unreadable data"
        body = text.strip("\n")
        return TextContent(f"[file: {filename or 'attachment'}]\n```\n{body}\n```")
    if kind == PDF_MEDIA_TYPE:
        return PdfContent(data=data, filename=filename or "document.pdf")
    image_type = cast("Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']", kind)
    return ImageContent(data=data, media_type=image_type)
