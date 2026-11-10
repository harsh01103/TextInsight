"""Sentiment analysis for English and Hinglish."""

from .api import sentiment
from .classical import ClassicalSentiment
from .hybrid import HybridSentiment
from .lexicon import LexiconSentiment
from .llm import LLMSentiment
from .onnx_backend import OnnxSentiment

__all__ = [
    "ClassicalSentiment",
    "HybridSentiment",
    "LLMSentiment",
    "LexiconSentiment",
    "OnnxSentiment",
    "sentiment",
]
