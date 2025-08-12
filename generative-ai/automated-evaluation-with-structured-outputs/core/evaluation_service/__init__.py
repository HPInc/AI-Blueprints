"""
Evaluation service package for automated evaluation with structured outputs.
"""

try:
    from .evaluation_model import EvaluationModel
    from .evaluation_service import EvaluationService
    __all__ = ["EvaluationModel", "EvaluationService"]
except ImportError:
    # Allow imports to fail during static analysis
    __all__ = []
