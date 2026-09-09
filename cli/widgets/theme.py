"""The look: one Textual theme — blue accents on the terminal's own
background — and nothing else.

The theme is an ANSI one (`ansi=True`): background and foreground are
`ansi_default`, the terminal's own, so the app sits in whatever the
person's terminal already is, light or dark; the accents are fixed
hexes chosen to read on either. Every colour a widget shows is quoted
from here by name — `[$primary]` in markup, `$primary` in CSS — so the
palette changes in one place. The primary is the prompt sign, the
borders, the headings; the secondary is the logo's second word and a
tool's own cards; the accent is the turn's account; success, warning
and error mark a tool's state.

Rules that follow from the background being unknown: never paint a
surface with a hex, never tint with an alpha (`$primary 25%` blends
against black, not the terminal); a highlighted row is a navy bar
(`$block-cursor-background`) with bright text on it, so a row's own
colours — a blue command name, a dim blurb — still read; a quiet border
is `$border-blurred`; secondary text is `$text-muted`, a grey that reads
on either background — never the terminal's `dim` attribute, which
fades to nothing in many terminals; the cursor is the terminal's own
reversed block.
"""

from __future__ import annotations

from textual.theme import Theme

VOID_THEME = Theme(
    name="void",
    primary="#5B9DFF",
    secondary="#6FD3E6",
    accent="#8FB8FF",
    warning="#E0B04D",
    error="#E5484D",
    success="#4CBF7A",
    dark=True,
    ansi=True,
    variables={
        "ansi-background": "ansi_default",
        "ansi-foreground": "ansi_default",
        "text-muted": "#A6B0BF",
        "text-disabled": "#7B8697",
        "border": "#5B9DFF",
        "border-blurred": "#3E5C8A",
        "block-cursor-background": "#26364F",
        "block-cursor-foreground": "#F4F7FB",
        "block-cursor-text-style": "none",
        "block-cursor-blurred-background": "#26364F",
        "block-cursor-blurred-foreground": "#F4F7FB",
        "block-cursor-blurred-text-style": "none",
        "block-hover-background": "#1D2A3F",
        "input-cursor-background": "ansi_default",
        "input-cursor-foreground": "ansi_default",
        "input-cursor-text-style": "reverse",
        "input-selection-background": "#3E5C8A",
        "input-selection-foreground": "#F4F7FB",
        "scrollbar": "#2A3441",
        "scrollbar-hover": "#3E5C8A",
        "scrollbar-active": "#5B9DFF",
        "scrollbar-background": "ansi_default",
        "scrollbar-background-hover": "ansi_default",
        "scrollbar-background-active": "ansi_default",
        "scrollbar-corner-color": "ansi_default",
        "link-color": "#5B9DFF",
        "link-color-hover": "#8FB8FF",
        "link-background-hover": "ansi_default",
    },
)
