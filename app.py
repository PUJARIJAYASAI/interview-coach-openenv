import gradio as gr
import os
from inference import run_simulation

def launch_app():
    with gr.Blocks(title="Interview Coach RL Environment") as demo:
        gr.Markdown("# 🎤 Interview Coach RL Environment")
        gr.Markdown("""
        Welcome to the **Interview Coach** Reinforcement Learning Environment. 
        This Space demonstrates how an AI agent (powered by **DeepSeek-R1**) learns to coach a candidate through strategic questioning and feedback.
        
        **Objective**: Maximize the candidate's learning while managing their fatigue.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                hf_token_input = gr.Textbox(
                    label="Hugging Face API Token",
                    placeholder="Enter your HF_TOKEN here...",
                    type="password",
                    value=os.getenv("HF_TOKEN", "")
                )
                task_dropdown = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="easy",
                    label="Select Task Difficulty"
                )
                run_btn = gr.Button("🚀 Start Simulation", variant="primary")
            
            with gr.Column(scale=2):
                log_output = gr.Code(
                    label="OpenEnv Logs",
                    language="markdown",
                    lines=20
                )
        
        gr.Markdown("""
        ### How it works:
        - **Rewards**: Improvement in candidate score (+), Guidance success (+), Repetition (-), Over-questioning (-).
        - **Logs**: Follows the strict **OpenEnv** standard for automated benchmarking.
        """)
        
        run_btn.click(
            fn=run_simulation,
            inputs=[task_dropdown, hf_token_input],
            outputs=log_output
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_app()
