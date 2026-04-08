import os
import sys
import time
import io
import contextlib
import traceback
from openai import OpenAI
from env import InterviewEnv
import grader

def run_simulation(task_name="easy", hf_token=None):
    """
    Runs the interview coach simulation and returns logs for display.
    """
    full_log = []
    
    try:
        # Configuration - Handle token precedence correctly
        api_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        model_name = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
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

        # Environment setup
        env = InterviewEnv()
        state = env.reset(task_name=task_name)
        
        # max_steps is already synced in env.reset() but we'll reflect it here for the log
        max_steps = env.max_steps

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
- Candidate Learning Factor: {state.get('learning_factor', 0.3)}
- Candidate Fatigue: {state.get('fatigue', 0)}

Valid Action Space (choose exactly one):
{valid_actions}

Strategy Guidelines:
1. Start with easy questions to build confidence.
2. Provide hints if the score is low.
3. Only advance difficulty when Learning Factor is stable.
4. Scale back if Fatigue exceeds 0.6.

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
                
                step_msg = f"[STEP] step={step_num} action={action} reward={reward:.2f} done={str(done).lower()} error=null"
                full_log.append(step_msg)
                
            except Exception as e:
                error_msg = str(e).replace("\n", " ").split("(")[0] # Truncate long error messages
                full_log.append(f"[STEP] step={step_num} action=error reward=0.00 done=true error={error_msg}")
                done = True
                break
                
        final_grade = grader.grade(state)
        success = final_grade >= 0.8
        rewards_str = ",".join([f"{r:.2f}" for r in total_rewards])
        # [END] must include: task, score, success, steps, rewards
        full_log.append(f"[END] task={task_name} score={final_grade:.2f} success={str(success).lower()} steps={len(total_rewards)} rewards={rewards_str}")

    except Exception as fatal_err:
        # Catch-all for unexpected framework/environment errors
        err_type = type(fatal_err).__name__
        full_log.append(f"[FATAL ERROR] {err_type}: {str(fatal_err)}")
        full_log.append(traceback.format_exc())

    return "\n".join(full_log)

if __name__ == "__main__":
    # Dynamically pick the task based on environment variables (used by the grader)
    task = os.getenv("TASK_NAME", "easy")
    print(run_simulation(task_name=task))
