"""
Evaluation service package for automated evaluation with structured outputs.
"""

# Make classes available but handle import errors gracefully
__all__ = ["EvaluationModel", "EvaluationService"]

def __getattr__(name):
    if name == "EvaluationModel":
        from .evaluation_model import EvaluationModel
        return EvaluationModel
    elif name == "EvaluationService":
        from .evaluation_service import EvaluationService
        return EvaluationService
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
