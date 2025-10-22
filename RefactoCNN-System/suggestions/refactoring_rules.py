
def suggest_refactoring(prediction_label, code_context):
    if prediction_label == 1:  # 1 indicates refactoring needed
        if "System.out.println" in code_context:
            return "Suggestion: Extract repeated print statements into a separate method."
        elif len(code_context.splitlines()) > 10:
            return "Suggestion: Method too long. Consider Extract Method refactoring."
        else:
            return "Suggestion: Refactor recommended. Review for possible improvements."
    else:
        return "No refactoring needed."
