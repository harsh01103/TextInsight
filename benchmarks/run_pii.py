"""Measure PII detection on a generated test set.

Usage: python benchmarks/run_pii.py [--count 2000] [--out FILE]

Every value is random and fake. Numbers get valid checksums so the rules can be
tested, and the set also has look-alikes that must not be flagged: card and Aadhaar
numbers with wrong checksums, order numbers, dates, prices and malformed PANs.

The set covers formats the rules were written for, so it checks that they work as
designed and says nothing about unusual formats in real text.
"""

from __future__ import annotations

import argparse
import json
import random
import string
from collections import Counter
from pathlib import Path

import textinsight as ti
from textinsight.pii.validators import luhn_valid, verhoeff_valid

SEED = 13
TEMPLATES = (
    "mera {label} hai {value}, please note kar lo",
    "My {label} is {value}.",
    "{label}: {value}",
    "Please contact me about this ({value}) today",
    "bhai yeh raha {value} check kar lena",
)


def digits(rng: random.Random, count: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(count))


def card(rng: random.Random) -> str:
    while True:
        number = rng.choice("3456") + digits(rng, 15)
        if luhn_valid(number) and len(set(number)) > 1:
            return " ".join(number[i : i + 4] for i in range(0, 16, 4))


def aadhaar(rng: random.Random) -> str:
    while True:
        number = rng.choice("23456789") + digits(rng, 11)
        if verhoeff_valid(number):
            return " ".join(number[i : i + 4] for i in range(0, 12, 4))


def phone(rng: random.Random) -> str:
    number = rng.choice("6789") + digits(rng, 9)
    return rng.choice([number, f"+91 {number}", f"+91-{number[:5]}-{number[5:]}", f"0{number}"])


def pan(rng: random.Random) -> str:
    letters = "".join(rng.choices(string.ascii_uppercase, k=3))
    return f"{letters}{rng.choice('ABCFGHJLPT')}{rng.choice(string.ascii_uppercase)}" + (
        digits(rng, 4) + rng.choice(string.ascii_uppercase)
    )


def email(rng: random.Random) -> str:
    name = "".join(rng.choices(string.ascii_lowercase, k=8))
    return f"{name}@{rng.choice(['example.com', 'mail.example.org', 'test.co.in'])}"


def upi(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=7)) + f"@{rng.choice(['okaxis', 'ybl'])}"


def url(rng: random.Random) -> str:
    return "https://example.com/" + "".join(rng.choices(string.ascii_lowercase, k=6))


GENERATORS = {
    "CARD": card,
    "AADHAAR": aadhaar,
    "PHONE": phone,
    "PAN": pan,
    "EMAIL": email,
    "UPI": upi,
    "URL": url,
}


def look_alike(rng: random.Random) -> str:
    """Return text with a number or code that is not personal information."""
    while True:
        bad_card = rng.choice("3456") + digits(rng, 15)
        bad_aadhaar = rng.choice("23456789") + digits(rng, 11)
        if not luhn_valid(bad_card) and not verhoeff_valid(bad_aadhaar):
            break
    return rng.choice(
        [
            f"card number was {bad_card}",
            f"id {bad_aadhaar} galat hai",
            f"order number {rng.choice('12345')}{digits(rng, 9)} ship ho gaya",
            f"date {rng.randint(1, 28):02d}-{rng.randint(1, 12):02d}-20{rng.randint(10, 26)} ko",
            f"price is Rs {rng.randint(100, 99999)}",
            f"code {''.join(rng.choices(string.ascii_uppercase, k=3))}Z{digits(rng, 6)} use karo",
            "yeh phone bahut accha hai",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rng = random.Random(SEED)
    expected: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    flagged: Counter[str] = Counter()
    for _ in range(args.count):
        for kind, generate in GENERATORS.items():
            value = generate(rng)
            template = rng.choice(TEMPLATES)
            text = template.format(label=kind.lower(), value=value)
            expected[kind] += 1
            for entity in ti.redact(text).entities:
                flagged[entity.kind] += 1
                correct[entity.kind] += entity.kind == kind and entity.value == value
        for entity in ti.redact(look_alike(rng)).entities:
            flagged[entity.kind] += 1

    report = {
        kind: {
            "precision": round(correct[kind] / flagged[kind], 4) if flagged[kind] else 0.0,
            "recall": round(correct[kind] / expected[kind], 4),
        }
        for kind in GENERATORS
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
