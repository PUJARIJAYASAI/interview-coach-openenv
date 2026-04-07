import os
import sys
import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import gradio as gr
from env import InterviewEnv
import grader
from inference import run_simulation

# --- FastAPI Setup ---
app = FastAPI(title="OpenEnv: Interview Coach")
env = InterviewEnv()

class ActionRequest(BaseModel):
    action: str

@app.post("/reset")
async def reset_env(request: Request):
    """
    OpenEnv Reset Endpoint. Returns the initial observation.
    """
    try:
        # Some graders pass a task name in JSON
        data = await request.json() if await request.body() else {}
        task_name = data.get("task_name", "easy")
    except:
        task_name = "easy"
        
    observation = env.reset(task_name=task_name)
    return {"observation": observation, "status": "initialized"}

@app.post("/step")
async def step_env(req: ActionRequest):
    """
    OpenEnv Step Endpoint. Takes an action and returns the transition.
    """
    valid_actions = env.get_valid_actions()
    if req.action not in valid_actions:
        # Fallback for LLM-based action strings
        action = "ask_easy_question"
        for va in valid_actions:
            if va in req.action:
                action = va
                break
    else:
        action = req.action

    observation, reward, done, info = env.step(action)
    return {
        "observation": observation,
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

@app.get("/state")
async def get_state():
    return env.state()

@app.get("/health")
async def health_check():
    return {"status": "ok", "benchmark": "interview_coach"}

# --- Gradio Web UI ---
def create_demo():
    with gr.Blocks(title="Interview Coach RL Dashboard") as demo:
        gr.Markdown("# 🎤 Interview Coach RL Environment")
        gr.Markdown("""
        Welcome to your **Interview Coach** Space. This dashboard allows you to run simulations manually 
        and view the **OpenEnv-compliant logs**.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                hf_token_input = gr.Textbox(
                    label="Hugging Face API Token",
                    placeholder="Enter HF_TOKEN or leave empty if Secret is set...",
                    type="password"
                )
                task_dropdown = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="easy",
                    label="Task Difficulty"
                )
                run_btn = gr.Button("🚀 Start Simulation", variant="primary")
            
            with gr.Column(scale=2):
                log_output = gr.Code(
                    label="OpenEnv Performance Logs",
                    language="markdown",
                    lines=20
                )
        
        run_btn.click(
            fn=run_simulation,
            inputs=[task_dropdown, hf_token_input],
            outputs=log_output
        )
    return demo

# --- Combine FastAPI & Gradio ---
demo = create_demo()
app = gr.mount_gradio_app(app, demo, path="/")

# Required for OpenEnv validation
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
