from interview_env import InterviewEnv

def make_grader(task_name, target_score):
    """
    Creates a grader function that evaluates a full trajectory.
    Score = 0.7 * (final answer score / 10) + 0.3 * (unique actions / total actions)
    """
    def grader(trajectory):
        if not trajectory:
            return 0.01
        
        # Get the last state in the trajectory
        final_state = trajectory[-1].get("state", {})
        final_answer_score = final_state.get("answer_score", 0)
        
        # Normalize final score to [0, 1]
        normalized_score = min(1.0, final_answer_score / 10.0)
        
        # Calculate action diversity
        actions = [step.get("action") for step in trajectory if step.get("action") is not None]
        if not actions:
            diversity_ratio = 0.0
        else:
            unique_actions = len(set(actions))
            diversity_ratio = unique_actions / len(actions)
            
        # Composite score calculation
        final_value = (0.7 * normalized_score) + (0.3 * diversity_ratio)
        
        # Clamp to (0, 1) as requested by OpenEnv Phase 2
        return max(0.01, min(0.99, final_value))
    
    return grader

def get_tasks():
    """
    Returns exactly 3 tasks as required by the platform.
    """
    tasks = []
    
    # Task 1: Easy
    tasks.append({
        "task_id": "easy",
        "task_name": "Easy Interview Prep",
        "description": "Help the candidate master basic Java and HR questions.",
        "env_class": InterviewEnv,
        "env_kwargs": {"task_name": "easy"},
        "grader": make_grader("easy", 0.6),
        "max_steps": 5
    })
    
    # Task 2: Medium
    tasks.append({
        "task_id": "medium",
        "task_name": "Technical Deep-Dive",
        "description": "Coach the candidate through medium-difficulty DSA and system concepts.",
        "env_class": InterviewEnv,
        "env_kwargs": {"task_name": "medium"},
        "grader": make_grader("medium", 0.7),
        "max_steps": 10
    })
    
    # Task 3: Hard
    tasks.append({
        "task_id": "hard",
        "task_name": "Senior Architect Challenge",
        "description": "Prepare the candidate for hard-level System Design and Architecture questions.",
        "env_class": InterviewEnv,
        "env_kwargs": {"task_name": "hard"},
        "grader": make_grader("hard", 0.8),
        "max_steps": 15
    })
    
    return tasks

TASKS = get_tasks()
