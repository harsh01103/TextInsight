"""Regular-expression fragments shared by the cleaner and the tokenizers.

Two details matter here:

* Python's ``\\w`` does not match combining marks, so a plain ``\\w+`` would cut
  Devanagari words apart at every vowel sign. ``WORD_CHARS`` adds them back.
* The patterns are anchored with lookbehinds where possible so that scanning a
  long input stays linear instead of retrying from every position.
"""

from __future__ import annotations

# Letters and digits, Latin combining marks, the Indic blocks (Devanagari to
# Sinhala) minus the danda punctuation, and the zero-width (non-)joiners.
WORD_CHARS = r"\ẁ-ͯऀ-ॣ०-෿‌‍"
DEVANAGARI_LETTERS = r"ऀ-ॣ०-ॿ"

URL = (
    r"(?:https?://|ftp://|www\.)"
    r"(?:\([^\s<>\"'()]*\)|[^\s<>\"'()])+"
    r"(?<![.,;:!?\]}])"
)
EMAIL = r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
MENTION = r"(?<![\w@])@\w+"
HASHTAG = rf"(?<![\w#])#[{WORD_CHARS}]+"

_EMOJI_BASE = r"\U0001F000-\U0001FAFF☀-➿⌀-⏿⬀-⯿"
_EMOJI_SUFFIX = r"[️\U0001F3FB-\U0001F3FF]?"
EMOJI = (
    r"[0-9#*]️?⃣"  # keycaps
    r"|[\U0001F1E6-\U0001F1FF]{2}"  # flags
    rf"|[{_EMOJI_BASE}]{_EMOJI_SUFFIX}(?:‍[{_EMOJI_BASE}]{_EMOJI_SUFFIX})*"  # joined sequences
)

EMOTICON = (
    r"(?<![\w:;=])"
    r"(?:[:;=][-'^]?[)(\]\[dDpPoO/\\|*3]"
    r"|[)(][-']?[:;=]"
    r"|<3|\^_\^|T_T|-_-|[xX][dD])"
    r"(?![\w])"
)

NUMBER = r"\d+(?:[.,:/-]\d+)*(?!\w)"
WORD = rf"[{WORD_CHARS}]+(?:['’\-][{WORD_CHARS}]+)*"
