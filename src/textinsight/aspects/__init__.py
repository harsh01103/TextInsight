"""Aspect-based sentiment for English and Hinglish."""

from .api import aspects
from .extractor import RuleAspects

__all__ = ["RuleAspects", "aspects"]
