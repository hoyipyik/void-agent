"""Emphasis that holds in text written without spaces: the parser a
`Reply` reads with.

CommonMark decides whether `**` may open or close by what stands beside
the run, and leans on the space between words to do it: a run with
punctuation on one side and a letter on the other neither opens nor
closes. Chinese, Japanese and Korean put no space there, so
`**氣溫：**18.3°C` — what a model answering in Chinese writes all day —
reaches the screen with its asterisks.

This is the CJK-friendly amendment to those rules
(github.com/tats-u/markdown-cjk-friendly): CJK punctuation never blocks a
run, and a CJK character beside one stands in for the space. Text with no
CJK beside the run reads exactly as CommonMark has it — `**Note:**text`
stays raw here as everywhere. Two things the amendment says are left out:
variation selectors are not looked behind, and a wide emoji counts as CJK,
since telling them apart takes a table Python does not carry and no reply
is read differently for it.

markdown-it keeps the decision in one place, `StateInline.scanDelims`,
which every rule that pairs markers asks (`emphasis`, `strikethrough`), so
that is all that is overridden; the rules themselves stay markdown-it's.
"""  # noqa: RUF002 — the fullwidth colon is the example

from __future__ import annotations

from unicodedata import east_asian_width, name

from markdown_it import MarkdownIt
from markdown_it.common.utils import isMdAsciiPunct, isPunctChar, isWhiteSpace
from markdown_it.parser_inline import ParserInline
from markdown_it.rules_inline.state_inline import Scanned, StateInline
from markdown_it.token import Token
from markdown_it.utils import EnvType


def parser() -> MarkdownIt:
    """Textual's own parser (`gfm-like`), its emphasis CJK-friendly."""
    md = MarkdownIt("gfm-like")
    inline = _Inline()
    # The preset's choice of rules lives in the rulers: they move over whole.
    inline.ruler, inline.ruler2 = md.inline.ruler, md.inline.ruler2
    md.inline = inline
    return md


def _cjk(char: str) -> bool:
    """Wide, fullwidth or halfwidth — or Hangul, whose conjoining jamo are
    neither."""
    return east_asian_width(char) in "WFH" or name(char, "").startswith("HANGUL")


def _punctuation(char: str) -> bool:
    return isMdAsciiPunct(ord(char)) or isPunctChar(char)


class _State(StateInline):
    def scanDelims(self, start: int, canSplitWord: bool) -> Scanned:
        """The run of markers at `start`: whether it may open, whether it
        may close, how long it is."""
        src = self.src
        end = start
        while end < self.posMax and src[end] == src[start]:
            end += 1
        # The edges of the text are whitespace, as CommonMark has them.
        before = src[start - 1] if start > 0 else " "
        after = src[end] if end < self.posMax else " "

        space_before, space_after = isWhiteSpace(ord(before)), isWhiteSpace(ord(after))
        cjk_before, cjk_after = _cjk(before), _cjk(after)
        punctuation_before, punctuation_after = _punctuation(before), _punctuation(after)
        # Only punctuation that is not CJK can block a run, and what it
        # needs on the other side is a space, punctuation — or a CJK
        # character.
        left_flanking = not space_after and (
            not punctuation_after or cjk_after or space_before or punctuation_before or cjk_before
        )
        right_flanking = not space_before and (
            not punctuation_before or cjk_before or space_after or punctuation_after or cjk_after
        )
        can_open = left_flanking and (canSplitWord or not right_flanking or punctuation_before)
        can_close = right_flanking and (canSplitWord or not left_flanking or punctuation_after)
        return Scanned(can_open, can_close, end - start)


class _Inline(ParserInline):
    def parse(self, src: str, md: MarkdownIt, env: EnvType, tokens: list[Token]) -> list[Token]:
        """`ParserInline.parse`, on the amended state — the one line of it
        that names the class."""
        state = _State(src, md, env, tokens)
        self.tokenize(state)
        for rule in self.ruler2.getRules(""):
            rule(state)
        return state.tokens
