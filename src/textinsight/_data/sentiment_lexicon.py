"""Sentiment word lists, grouped by strength.

Hinglish keys must use canonical spellings from ``_data/hinglish.py`` wherever a
canonical form exists, because tokens are normalized before lookup.
"""

from __future__ import annotations


def _table(*groups: tuple[float, str]) -> dict[str, float]:
    return {word: value for value, words in groups for word in words.split()}


ENGLISH = _table(
    (
        3.2,
        """amazing awesome excellent fantastic outstanding superb wonderful brilliant perfect
        phenomenal incredible magnificent marvelous exceptional love loved loving adore""",
    ),
    (
        2.2,
        """good great nice beautiful best happy glad delighted pleased enjoy enjoyed enjoying
        lovely fabulous sweet proud winner win won congratulations congrats blessed thanks
        thank grateful recommend recommended impressive impressed friendly helpful bless
        wishes super""",
    ),
    (
        1.3,
        """fine okay ok cool fair decent like liked likes fast quick clean comfortable easy
        useful smart strong safe fresh worth better improved support hope hopeful respect
        kind cute fun funny interesting glad peace pretty smooth reliable wish luck""",
    ),
    (
        -1.3,
        """slow late expensive costly problem problems issue issues difficult hard boring
        confused confusing weak dirty sad sorry miss lack lacking fail poor cheap unfair
        hurt tired worried concern ignore ignored delay delayed""",
    ),
    (
        -2.2,
        """bad wrong hate hated angry annoying annoyed broken damaged failed fake rude stupid
        useless waste wasted disappointed disappointing disappointment corrupt liar lies
        fool idiot shame shameful nonsense sucks trouble unhappy refused scam fraud cheat""",
    ),
    (
        -3.2,
        """terrible horrible awful worst disgusting pathetic hopeless nightmare garbage
        rubbish trash dreadful despise""",
    ),
)

HINGLISH = _table(
    (
        3.2,
        """zabardast shandaar behtareen mast khubsurat lajawab kamaal dhansu umda mahan""",
    ),
    (
        2.2,
        """accha acche acchi badhiya pyaar pyara pyari khush sundar dhanyavaad shukriya
        mubarak badhai jai subhkamnaye shubhkamnaye dua shubh""",
    ),
    (
        1.3,
        """sahi thik sasta sasti maza aasan pasand jeet hardik sukoon""",
    ),
    (
        -1.3,
        """mehenga mehengi dikkat pareshan mushkil dukhi nirash galat dard takleef""",
    ),
    (
        -2.2,
        """bura buri bure bekaar kharab ganda gandi gussa dhoka barbaad jhooth chor faltu
        paagal sharam besharam nalayak lanat nafrat gaddar neech kutta kutte haram""",
    ),
    (
        -3.2,
        """bakwas ghatiya harami kamina nikamma""",
    ),
)

EMOJI = _table(
    (2.5, "😀 😃 😄 😁 😆 😊 😍 🥰 😘 😎 🤩 👍 👏 🙌 ❤ ❤️ 💕 💖 💯 🎉 🔥"),
    (1.5, "🙂 😉 😂 🤣 🙏"),
    (-1.5, "😕 😟 😞 😔 😢"),
    (-2.5, "😠 😡 🤬 😭 👎 💔 🤮 😤"),
)

EMOTICONS = _table(
    (2.0, ":) :-) :D :-D ;) ;-) =) =D <3 xD XD ^_^"),
    (-2.0, ":( :-( :'( :/ :-/ =( T_T -_-"),
)

NEGATIONS_BEFORE = frozenset(
    """not no never nor without hardly cannot can't don't doesn't didn't won't isn't aren't
    wasn't weren't wouldn't shouldn't couldn't haven't hasn't hadn't nahi na mat bina""".split()
)
# Hindi usually places the negation after the word: "accha nahi hai".
NEGATIONS_AFTER = frozenset({"nahi", "na"})

INTENSIFIERS = _table(
    (0.3, "very really extremely absolutely totally so too bahut zyada ekdum bilkul itna"),
    (-0.3, "slightly somewhat barely thoda kam"),
)

CONTRAST = frozenset({"but", "however", "lekin", "magar"})
