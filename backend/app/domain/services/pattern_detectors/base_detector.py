"""
Base pattern detector interface.
Defines contract for all pattern detection implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class PatternDetector(ABC):
    """
    Abstract base class for pattern detectors.
    Each detector identifies specific algorithmic patterns in AST.
    """

    @abstractmethod
    def detect(self, node: Any) -> Dict[str, Any]:
        """
        Detect specific patterns in the given AST node.
        """
