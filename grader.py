def grade_easy(state):
    return min(1.0, state.get("answer_score", 0) / 8)

def grade_medium(state):
    return min(1.0, state.get("answer_score", 0) / 9)

def grade_hard(state):
    # harder scoring
    base = state.get("answer_score", 0) / 10
    
    # bonus for consistency (using confidence/learning_factor)
    if state.get("confidence", 0) > 0.7:
        base += 0.1

    return min(1.0, base)

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
