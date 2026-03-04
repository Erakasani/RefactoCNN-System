from __future__ import annotations
from typing import Dict, List
from .rules import detect_long_method, detect_duplication, detect_naming_issues

def map_prediction_to_suggestion(pred_label: int, prob_refactor: float, segment: Dict, tokens: List[str]) -> Dict:
    """Map model prediction to an actionable refactoring suggestion.

    Returns a dict with:
      - suggestion: str
      - confidence: float (probability aligned with predicted class)
      - rules_fired: List[str]
      - reason: str (short explanation)
    """
    rules_fired: List[str] = []
    code = segment.get("code", "")

    if pred_label == 0:
        conf = float(1.0 - prob_refactor)
        return {
            "suggestion": "No Refactoring Needed",
            "confidence": conf,
            "rules_fired": rules_fired,
            "reason": "Model confidence indicates the segment is maintainable without refactoring.",
        }

    # pred_label == 1
    is_long = detect_long_method(code)
    is_dup = detect_duplication(tokens)
    is_name = detect_naming_issues(tokens)

    if is_long: rules_fired.append("long_method")
    if is_dup: rules_fired.append("duplication")
    if is_name: rules_fired.append("naming_issues")

    if is_long and is_dup:
        suggestion = "Extract Method"
        reason = "Detected a long method with repeated token patterns; extraction can improve readability and reuse."
    elif is_dup:
        suggestion = "Unify Duplicate Code"
        reason = "Detected repeated token patterns; consolidating duplicates can reduce maintenance cost."
    elif is_name:
        suggestion = "Rename Variable"
        reason = "Detected short or non-descriptive identifier patterns; renaming can improve clarity."
    else:
        suggestion = "General Refactor Recommended"
        reason = "Model predicts refactoring is beneficial; apply standard cleanup to improve structure and readability."

    return {
        "suggestion": suggestion,
        "confidence": float(prob_refactor),
        "rules_fired": rules_fired,
        "reason": reason,
    }
