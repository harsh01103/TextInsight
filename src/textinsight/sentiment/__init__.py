"""Sentiment analysis for English and Hinglish."""

from .api import sentiment
from .classical import ClassicalSentiment
from .hybrid import HybridSentiment
from .lexicon import LexiconSentiment

__all__ = ["ClassicalSentiment", "HybridSentiment", "LexiconSentiment", "sentiment"]
