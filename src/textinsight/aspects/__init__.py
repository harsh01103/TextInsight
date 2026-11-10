"""Aspect-based sentiment for English and Hinglish."""

from .api import aspects
from .extractor import RuleAspects
from .llm import LLMAspects

__all__ = ["LLMAspects", "RuleAspects", "aspects"]
