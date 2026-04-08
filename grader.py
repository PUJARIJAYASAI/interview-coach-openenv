def clamp_score(score):
    """
    OpenEnv Phase 2 requires scores to be strictly between 0 and 1.
    This clamps values to [0.01, 0.99].
    """
    return max(0.01, min(0.99, score))

def grade_easy(state):
    raw = state.get("answer_score", 0) / 8
    return clamp_score(raw)

def grade_medium(state):
    raw = state.get("answer_score", 0) / 9
    return clamp_score(raw)

def grade_hard(state):
    # harder scoring
    base = state.get("answer_score", 0) / 10
    
    # bonus for consistency (using confidence/learning_factor)
    if state.get("confidence", 0) > 0.7:
        base += 0.1

    return clamp_score(base)

def grade(state):
    """
    Main grader entry point. Normalizes performance to 0.0-1.0.
    Selects task-specific logic based on the state.
    """
    task = state.get("task_name", "easy")
    if task == "hard":
        return grade_hard(state)
    elif task == "medium":
        return grade_medium(state)
    else:
        return grade_easy(state)
