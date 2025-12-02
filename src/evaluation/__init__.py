# Evaluation module
from .gt_loader import GTLoader, GTObject, GTFrame
from .evaluators import (
    DetectionEvaluator,
    TrackingEvaluator,
    LocalizationEvaluator,
    DetectionMatch,
)
from .matching import compute_iou, match_detections, compute_euclidean_distance
from .runner import (
    SequenceResult,
    EvaluationResult,
    SequenceEvaluator,
    MultiSequenceEvaluator,
)
