import os
import sys
import time
from openai import OpenAI
from env import InterviewEnv
import grader

def run_simulation():
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    TASK_NAME = os.getenv("TASK_NAME", "easy")
    BENCHMARK = "interview_coach"
    
    if not API_KEY:
        print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")
        print(f"[STEP] step=1 action=none reward=0.00 done=true error=Missing HF_TOKEN")
        print(f"[END] success=false steps=1 rewards=0.00")
        return

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")
        print(f"[STEP] step=1 action=none reward=0.00 done=true error=Init error: {str(e)}")
        print(f"[END] success=false steps=1 rewards=0.00")
        return

    env = InterviewEnv()
    state = env.reset(task_name=TASK_NAME)
    
    step_limits = {"easy": 5, "medium": 10, "hard": 15}
    env.max_steps = step_limits.get(TASK_NAME, 8)

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")
    
    step = 0
    rewards = []
    done = False
    
    while not done:
        step += 1
        valid_actions = env.get_valid_actions()

        prompt = f"""
Role: Interview Coach AI
Goal: Maximize candidate improvement.

Current State:
- Question: {state['question']}
- Score: {state['answer_score']}/10
- Learning Factor: {state.get('learning_factor', 0.3)}
- Fatigue: {state.get('fatigue', 0)}

Valid Actions:
{valid_actions}

Return ONLY the exact action name.
"""
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=20
            )
            raw_action = response.choices[0].message.content.strip().lower()
            
            action = "ask_easy_question"
            for va in ["ask_easy_question", "ask_medium_question", "ask_hard_question", "give_hint", "give_feedback", "change_topic"]:
                if va in raw_action:
                    action = va
                    break
            
            state, reward, done, _ = env.step(action)
            rewards.append(f"{reward:.2f}")
            error = None
            
            print(f"[STEP] step={step} action={action} reward={reward:.2f} done={'true' if done else 'false'} error={error if error else 'null'}")
            
        except Exception as e:
            error_msg = str(e).replace("\n", " ")
            print(f"[STEP] step={step} action=error reward=0.00 done=true error={error_msg}")
            done = True
            break
            
    final_grade = grader.grade(state)
    success = final_grade >= 0.8
    print(f"[END] success={'true' if success else 'false'} steps={len(rewards)} rewards={','.join(rewards)}")

if __name__ == "__main__":
    run_simulation()
