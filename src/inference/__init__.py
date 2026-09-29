"""mineRakshak-ai inference package.

Provides production inference pipeline, model loading, input validation,
and real-time collision risk assessment for autonomous haul truck operations.
"""

from src.inference.predict import (
    InferenceValidationError,
    MineRakshakInferenceEngine,
    RESPONSE_MAPPINGS,
    SAFETY_DISCLAIMER,
    get_inference_engine,
    predict,
)

__all__ = [
    "MineRakshakInferenceEngine",
    "InferenceValidationError",
    "get_inference_engine",
    "predict",
    "RESPONSE_MAPPINGS",
    "SAFETY_DISCLAIMER",
]
