"""Checksum and format rules for Indian identifiers and payment cards."""

from __future__ import annotations

__all__ = ["is_valid_aadhaar", "is_valid_card", "is_valid_pan", "luhn_valid", "verhoeff_valid"]

_VERHOEFF_MULTIPLY = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_PERMUTE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_PAN_HOLDER_TYPES = frozenset("ABCFGHJLPT")


def _digits(value: str) -> str:
    return "".join(char for char in value if char.isdigit())


def luhn_valid(number: str) -> bool:
    """Check the Luhn checksum used by payment cards."""
    digits = _digits(number)
    if not digits:
        return False
    total = 0
    for position, char in enumerate(reversed(digits)):
        digit = int(char)
        if position % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def verhoeff_valid(number: str) -> bool:
    """Check the Verhoeff checksum used by Aadhaar numbers."""
    digits = _digits(number)
    if not digits:
        return False
    check = 0
    for position, char in enumerate(reversed(digits)):
        check = _VERHOEFF_MULTIPLY[check][_VERHOEFF_PERMUTE[position % 8][int(char)]]
    return check == 0


def is_valid_card(number: str) -> bool:
    """Check length and Luhn, and reject repeated digits such as all zeros."""
    digits = _digits(number)
    return 13 <= len(digits) <= 19 and len(set(digits)) > 1 and luhn_valid(digits)


def is_valid_aadhaar(number: str) -> bool:
    """Check that a number has 12 digits, does not start with 0 or 1, and passes Verhoeff."""
    digits = _digits(number)
    return len(digits) == 12 and digits[0] not in "01" and verhoeff_valid(digits)


def is_valid_pan(value: str) -> bool:
    """Check the PAN format: 5 letters, 4 digits, 1 letter, with a known holder type."""
    pan = value.upper()
    return (
        len(pan) == 10
        and pan[:5].isalpha()
        and pan[5:9].isdigit()
        and pan[9].isalpha()
        and pan[3] in _PAN_HOLDER_TYPES
    )
