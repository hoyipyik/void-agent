"""The model-facing projection with attachments intact: a persisted parts
array as the ordered content the MODEL receives.

Text speaks exactly as `context_text` renders it; each `file` part — the
attachment a client sends — becomes the `ImageContent` or `PdfContent` it
carries, in send order, so a message built from these parts is what
`Message.user(...)` would have built by hand; a text file joins the text
around it, fenced under its filename. An attachment that cannot be sent
(a media type no provider takes, unreadable data) is named in the text
instead, so the model knows what it did not receive."""

from __future__ import annotations

from typing import Any

from void_agent.core.content import ContentPart, TextContent
from void_agent.core.parts.attachment import attachment_label, decode_attachment
from void_agent.core.parts.context import part_line


def context_content(parts: list[dict[str, Any]]) -> tuple[ContentPart, ...]:
    """A persisted parts array as the content the MODEL receives: the lines
    of `context_text`, grouped into one text part between the attachments
    they surround, and each attachment as its own content part. Empty when
    nothing speaks — the caller leaves such a message out."""
    content: list[ContentPart] = []
    lines: list[str] = []

    def flush() -> None:
        if lines:
            content.append(TextContent("\n".join(lines)))
            lines.clear()

    for part in parts:
        if part.get("type") == "file":
            decoded = decode_attachment(part)
            if isinstance(decoded, str):
                lines.append(f"[{attachment_label(part)}, not sent: {decoded}]")
            elif isinstance(decoded, TextContent):
                lines.append(decoded.text)
            else:
                flush()
                content.append(decoded)
        elif (line := part_line(part)) is not None:
            lines.append(line)
    flush()
    return tuple(content)
