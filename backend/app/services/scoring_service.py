"""Scoring engine for the AI/ML Validation Platform.

Calculates the Platform Model Health Score based on validator results.

Scoring Formula:
    Overall Score = Weighted average of category scores
    
    Default weights (configurable):
        performance:     1.0
        data_quality:    1.0
        fairness:        1.0
        drift:           1.0
        robustness:      1.0
        explainability:  1.0
        reliability:     1.0
    
    Each category score ranges from 0 to 100.
    
    IMPORTANT: This score is a platform-specific assessment based on
    configurable rules. It is NOT an industry-standard certification.
"""
import logging
from typing import Any
from app.validators.base import ValidatorResult, ValidatorStatus

logger = logging.getLogger('ai_validation')

# Default category weights (all equal)
# Can be overridden via validation run config
DEFAULT_WEIGHTS: dict[str, float] = {
    'performance': 1.0,
    'data_quality': 1.0,
    'fairness': 1.0,
    'drift': 1.0,
    'robustness': 1.0,
    'explainability': 1.0,
    'reliability': 1.0,
}


def calculate_scores(
    results: list[ValidatorResult],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Calculate category scores and overall Model Health Score.
    
    Args:
        results: List of ValidatorResult from all validators.
        weights: Optional custom weights per category.
    
    Returns:
        Dictionary with 'overall' score and 'categories' dict.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Group results by category
    category_results: dict[str, list[ValidatorResult]] = {}
    for result in results:
        cat = getattr(result, 'category', None)
        if not cat:
            cat = getattr(result, 'metadata', {}).get('category', None)
        if not cat:
            for known_cat in weights.keys():
                if known_cat in result.validator_name.lower():
                    cat = known_cat
                    break
        if not cat:
            cat = 'general'
        
        if cat not in category_results:
            category_results[cat] = []
        category_results[cat].append(result)
    
    # Calculate per-category scores
    category_scores: dict[str, float] = {}
    for category, cat_results in category_results.items():
        scores = [
            r.score for r in cat_results
            if r.score is not None and r.status not in (ValidatorStatus.SKIPPED, ValidatorStatus.ERROR)
        ]
        if scores:
            category_scores[category] = sum(scores) / len(scores)
    
    # Calculate overall weighted score
    total_weight = 0.0
    weighted_sum = 0.0
    for category, score in category_scores.items():
        w = weights.get(category, 1.0)
        weighted_sum += score * w
        total_weight += w
    
    overall = weighted_sum / total_weight if total_weight > 0 else None
    
    return {
        'overall': round(overall, 4) if overall is not None else None,
        'categories': {k: round(v, 4) for k, v in category_scores.items()},
    }
