import gradio as gr
import os
from inference import run_simulation

def launch_app():
    # Detect if a secret is available to provide better UI feedback
    HAS_SECRET = "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"].strip() != ""
    
    with gr.Blocks(title="Interview Coach RL Environment") as demo:
        gr.Markdown("# 🎤 Interview Coach RL Environment")
        gr.Markdown("""
        Welcome to the **Interview Coach** Reinforcement Learning Environment. 
        This Space demonstrates how an AI agent (powered by **DeepSeek-R1**) learns to coach a candidate through strategic questioning and feedback.
        
        **Objective**: Maximize the candidate's learning while managing their fatigue.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                token_label = "Hugging Face API Token"
                if HAS_SECRET:
                    token_label += " (Optional - Secret Detected ✅)"
                
                hf_token_input = gr.Textbox(
                    label=token_label,
                    placeholder="Enter token OR leave empty to use Space Secrets...",
                    type="password",
                    value=""
                )
                
                task_dropdown = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="easy",
                    label="Select Task Difficulty"
                )
                
                run_btn = gr.Button("🚀 Start Simulation", variant="primary")
                
                if HAS_SECRET:
                    gr.Markdown("> [!TIP]\n> **Secret Detected**: You don't need to enter a token manually. The Space will use your configured `HF_TOKEN` Secret.")
                else:
                    gr.Markdown("> [!IMPORTANT]\n> **No Secret Found**: Please provide a token in the box above or add an `HF_TOKEN` Secret in Settings.")
            
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
        
        def run_wrapper(task, token):
            # Show a processing message before simulation starts
            return run_simulation(task, token)

        run_btn.click(
            fn=run_wrapper,
            inputs=[task_dropdown, hf_token_input],
            outputs=log_output
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_app()
