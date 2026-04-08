import logging

# Configure logging to help debug automated validation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openenv_grader")

def clamp_score(score):
    """
    OpenEnv Phase 2 requires scores to be strictly between 0 and 1.
    This clamps values to [0.01, 0.99] to handle edge cases.
    """
    try:
        val = float(score)
        clamped = max(0.01, min(0.99, val))
        return clamped
    except Exception:
        return 0.05

def grade_easy(state):
    try:
        score = min(1.0, state.get("answer_score", 0) / 8)
        logger.info(f"Grading Easy: {score}")
        return clamp_score(score)
    except Exception as e:
        logger.error(f"Error in grade_easy: {e}")
        return 0.05

def grade_medium(state):
    try:
        score = min(1.0, state.get("answer_score", 0) / 9)
        logger.info(f"Grading Medium: {score}")
        return clamp_score(score)
    except Exception as e:
        logger.error(f"Error in grade_medium: {e}")
        return 0.05

def grade_hard(state):
    try:
        # harder scoring
        base = state.get("answer_score", 0) / 10
        
        # bonus for consistency (using confidence/learning_factor)
        if state.get("confidence", 0) > 0.7:
            base += 0.1

        score = min(1.0, base)
        logger.info(f"Grading Hard: {score}")
        return clamp_score(score)
    except Exception as e:
        logger.error(f"Error in grade_hard: {e}")
        return 0.05

def grade(state):
    """
    Main grader entry point. Normalizes performance to 0.0-1.0.
    Selects task-specific logic based on the state.
    """
    try:
        task = state.get("task_name", "easy")
        if task == "hard":
            return grade_hard(state)
        elif task == "medium":
            return grade_medium(state)
        else:
            return grade_easy(state)
    except Exception:
        return 0.05
