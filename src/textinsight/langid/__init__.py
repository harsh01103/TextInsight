"""Language identification for English, Hindi and code-mixed Hinglish."""

from .detector import HeuristicLanguageDetector, detect_language
from .ngram import CharNgramClassifier

__all__ = ["CharNgramClassifier", "HeuristicLanguageDetector", "detect_language"]
