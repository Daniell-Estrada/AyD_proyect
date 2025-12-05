"""
Pattern detection utilities package.
Contains detectors for various algorithmic patterns and paradigms.
"""

from app.domain.services.pattern_detectors.backtracking_detector import BacktrackingPatternDetector
from app.domain.services.pattern_detectors.base_detector import PatternDetector
from app.domain.services.pattern_detectors.divide_conquer_detector import DivideAndConquerPatternDetector
from app.domain.services.pattern_detectors.greedy_detector import GreedyPatternDetector
from app.domain.services.pattern_detectors.loop_detector import LoopPatternDetector
from app.domain.services.pattern_detectors.memoization_detector import MemoizationPatternDetector
from app.domain.services.pattern_detectors.recursion_detector import RecursionPatternDetector

__all__ = [
    "PatternDetector",
    "RecursionPatternDetector",
    "DivideAndConquerPatternDetector",
    "MemoizationPatternDetector",
    "GreedyPatternDetector",
    "BacktrackingPatternDetector",
    "LoopPatternDetector",
]
