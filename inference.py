import os
import sys
import time
import io
import contextlib
import traceback
from openai import OpenAI
from interview_env import InterviewEnv
from tasks import TASKS

def run_simulation(task_name="easy", hf_token=None):
    """
    Runs the interview coach simulation using the trajectory-based TASKS manifest.
    """
    full_log = []
    trajectory = []
    
    # Discovery: Find the correct task dict
    task_config = next((t for t in TASKS if t["task_id"] == task_name), TASKS[0])
    
    try:
        # Configuration - Handle token precedence correctly
        api_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        env_name = "interview_coach"
        
        # Use provided token, then fallback to environment variables
        api_key = hf_token if hf_token and hf_token.strip() else os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        
        if not api_key:
            start_msg = f"[START] task={task_name} env={env_name} model={model_name}"
            step_msg = f"[STEP] step=1 action=none reward=0.00 done=true error=Missing Token. Please provide HF_TOKEN in the UI or as a Space Secret."
            end_msg = f"[END] task={task_name} score=0.01 success=false steps=1 rewards=0.00"
            return "\n".join([start_msg, step_msg, end_msg])

        try:
            client = OpenAI(base_url=api_url, api_key=api_key)
        except Exception as client_err:
            start_msg = f"[START] task={task_name} env={env_name} model={model_name}"
            step_msg = f"[STEP] step=1 action=none reward=0.00 done=true error=Client Initialization Error: {str(client_err)}"
            end_msg = f"[END] task={task_name} score=0.01 success=false steps=1 rewards=0.00"
            return "\n".join([start_msg, step_msg, end_msg])

        # Environment setup from TASKS manifest
        env = task_config["env_class"](**task_config["env_kwargs"])
        state = env.reset(task_name=task_name)
        
        max_steps = task_config["max_steps"]
        full_log.append(f"[START] task={task_name} env={env_name} model={model_name}")
        
        step_num = 0
        total_rewards = []
        done = False
        
        while not done:
            step_num += 1
            valid_actions = env.get_valid_actions()

            prompt = f"""
Role: Expert Interview Coach AI
Goal: Strategically improve the candidate's performance using Reinforcement Learning.

Current Session Meta:
- Task Complexity: {task_name}
- Step: {step_num}/{max_steps}

Environment Observation:
- Current Question: {state['question']}
- Difficulty: {state['difficulty']}
- Current Answer Score: {state['answer_score']}/10

Valid Action Space:
{valid_actions}

Return ONLY the action name from the list.
"""
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=64
                )
                content = response.choices[0].message.content
                raw_action = content.strip().lower() if content else ""
                
                # Map response to valid actions
                action = "ask_easy_question"
                for va in ["ask_easy_question", "ask_medium_question", "ask_hard_question", "give_hint", "give_feedback", "change_topic"]:
                    if va in raw_action:
                        action = va
                        break
                
                state, reward, done, _ = env.step(action)
                total_rewards.append(reward)
                
                # Log step
                step_msg = f"[STEP] step={step_num} action={action} reward={reward:.2f} done={str(done).lower()} error=null"
                full_log.append(step_msg)
                
                # Record trajectory for the grader
                trajectory.append({
                    "step": step_num,
                    "action": action,
                    "state": state,
                    "reward": reward
                })
                
                if step_num >= max_steps:
                    done = True
                
            except Exception as e:
                error_msg = str(e).replace("\n", " ").split("(")[0]
                full_log.append(f"[STEP] step={step_num} action=error reward=0.00 done=true error={error_msg}")
                done = True
                break
                
        # Final Grade using the trajectory grader from TASKS
        final_grade = task_config["grader"](trajectory)
        success = final_grade >= 0.7 # threshold for Success
        rewards_str = ",".join([f"{r:.2f}" for r in total_rewards])
        full_log.append(f"[END] task={task_name} score={final_grade:.2f} success={str(success).lower()} steps={len(total_rewards)} rewards={rewards_str}")

    except Exception as fatal_err:
        err_type = type(fatal_err).__name__
        full_log.append(f"[FATAL ERROR] {err_type}: {str(fatal_err)}")
        full_log.append(traceback.format_exc())

    return "\n".join(full_log)

if __name__ == "__main__":
    task = os.getenv("TASK_NAME", "easy")
    print(run_simulation(task_name=task))
