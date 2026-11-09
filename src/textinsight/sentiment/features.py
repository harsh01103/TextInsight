"""Text features for the trained classifier.

Training and inference call this same code, so they cannot drift apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .._data.sentiment_lexicon import NEGATIONS_AFTER, NEGATIONS_BEFORE
from ..langid.detector import detect_language
from ..preprocess.cleaner import clean
from ..preprocess.hinglish import canonical_form, phonetic_key, squeeze_repeats
from ..preprocess.normalizer import reduce_elongation
from ..preprocess.tokenizer import iter_tokens
from .lexicon import score_text

__all__ = ["FeatureOptions", "extract_features", "feature_tokens"]

_KEPT_KINDS = {"word", "emoji", "emoticon"}
_NGRAM_PREFIX = {1: "w", 2: "b", 3: "t"}
_NEGATION_SCOPE_AFTER = 3
_NEGATION_SCOPE_BEFORE = 2
_SCOPE_BREAKERS = {"!", "?"}
# Tweets cut off by the platform end in a partial word followed by an ellipsis.
_TRUNCATED_END = re.compile(r"\S+\s*(?:…|\.\.\.)\s*$")
_RETWEET_MARK = "rt"
_VOWELS = re.compile("[aeiou]")
_MIN_SKELETON_WORD = 2
_CODE_MIX_BINS = 4
# Religion, nationality and party words lean negative in the training tweets. Masking them
# stops the model from learning that a group of people is itself negative.
_IDENTITY_TERMS = frozenset(
    """hindu hindus hinduo muslim muslims musalman mulla islam islamic sikh sikhs christian
    christians dalit dalits pakistan pakistani pakistanis bjp congress""".split()
)
_IDENTITY_MASK = "<identity>"


@dataclass(frozen=True)
class FeatureOptions:
    """Which feature families to generate. Saved with each trained model."""

    hinglish: bool = True
    word_ngrams: tuple[int, ...] = (1, 2)
    char_sizes: tuple[int, ...] = (3, 4)
    negation: bool = False
    lexicon: bool = False
    raw_words: bool = False
    drop_noise: bool = False
    phonetic: bool = False
    skeleton: bool = False
    code_mix: bool = False
    mask_identity: bool = False


DEFAULT_OPTIONS = FeatureOptions()


def feature_tokens(text: str, *, hinglish: bool = True) -> list[str]:
    """Reduce text to the tokens used for features."""
    cleaned = clean(text, lowercase=True, mentions=True, numbers=True, hashtags="strip_symbol")
    tokens: list[str] = []
    for token, kind in iter_tokens(reduce_elongation(cleaned)):
        if kind in _KEPT_KINDS:
            tokens.append(canonical_form(token) if hinglish and kind == "word" else token)
        elif token in _SCOPE_BREAKERS:
            tokens.append(token)
    return tokens


def _negated_words(tokens: list[str]) -> set[str]:
    """Words in the scope of a negation: "accha nahi" and "not good" both mark the adjective."""
    marked: set[str] = set()
    for index, token in enumerate(tokens):
        if token in NEGATIONS_BEFORE:
            for following in tokens[index + 1 : index + 1 + _NEGATION_SCOPE_AFTER]:
                if following in _SCOPE_BREAKERS:
                    break
                marked.add(following)
        if token in NEGATIONS_AFTER:
            marked.update(tokens[max(0, index - _NEGATION_SCOPE_BEFORE) : index])
    return marked


def _masked(word: str, options: FeatureOptions) -> bool:
    return options.mask_identity and word in _IDENTITY_TERMS


def _skeleton(word: str) -> str:
    """Drop later vowels, because Hinglish writers often omit them: bahut, bht and bohot match."""
    squeezed = squeeze_repeats(word)
    return squeezed[0] + _VOWELS.sub("", squeezed[1:])


def _code_mix_features(text: str) -> list[str]:
    """Describe how much of the text is Hindi versus English."""
    result = detect_language(text)
    hindi = result.token_languages.count("hi")
    english = result.token_languages.count("en")
    words = hindi + english
    share = round(hindi / words * _CODE_MIX_BINS) if words else 0
    return [f"mix:lang_{result.language}", f"mix:hi_share_{share}"]


def extract_features(text: str, options: FeatureOptions = DEFAULT_OPTIONS) -> list[str]:
    """Return the features for ``text`` under ``options``."""
    if options.drop_noise:
        text = _TRUNCATED_END.sub("", text)
    tokens = feature_tokens(text, hinglish=options.hinglish)
    if options.drop_noise:
        tokens = [t for t in tokens if t != _RETWEET_MARK]
    if options.mask_identity:
        tokens = [_IDENTITY_MASK if t in _IDENTITY_TERMS else t for t in tokens]

    features: list[str] = []
    for size in options.word_ngrams:
        prefix = _NGRAM_PREFIX[size]
        features += [
            f"{prefix}:{' '.join(tokens[i : i + size])}" for i in range(len(tokens) - size + 1)
        ]
    for token in tokens:
        if token.isalpha():
            marked = f"<{token}>"
            for size in options.char_sizes:
                features += [f"c:{marked[i : i + size]}" for i in range(len(marked) - size + 1)]
    if options.negation:
        features += [f"n:{word}" for word in _negated_words(tokens)]
    if options.raw_words:
        features += [f"r:{word}" for word in feature_tokens(text, hinglish=False)]
    if options.phonetic or options.skeleton:
        raw = [
            word
            for word in feature_tokens(text, hinglish=False)
            if word.isascii() and word.isalpha() and not _masked(word, options)
        ]
        if options.phonetic:
            features += [f"p:{phonetic_key(word)}" for word in raw]
        if options.skeleton:
            features += [f"s:{_skeleton(word)}" for word in raw if len(word) > _MIN_SKELETON_WORD]
    if options.code_mix:
        features += _code_mix_features(text)
    if options.lexicon:
        score, label = score_text(text)
        features += [f"lex:{label}", f"lex:bin{round(score * 2)}"]
    return features
