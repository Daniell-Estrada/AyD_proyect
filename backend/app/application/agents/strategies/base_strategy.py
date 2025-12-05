"""
Base strategy interface for complexity analysis.
Defines the contract that all analysis strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class ComplexityAnalysisStrategy(ABC):
    """
    Abstract base class for complexity analysis strategies.
    Implements Strategy Pattern to enable different analysis techniques
    for various algorithmic paradigms.
    """

    @abstractmethod
    def analyze(
        self, ast_dict: Dict[str, Any], patterns: Dict[str, Any]
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, str]]:
        """
        Analyze the algorithm represented by the AST and detected patterns.
        """
