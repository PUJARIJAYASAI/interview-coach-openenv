🚀 Interview Coach RL Environment
🧠 Overview

Interview Coach is a Reinforcement Learning environment where an AI agent learns how to conduct effective technical interviews.

Instead of solving questions, the agent focuses on strategy — deciding when to ask questions, give hints, or provide feedback to improve a candidate’s performance.

🎯 Motivation

Real interviews are dynamic. A good interviewer adapts based on the candidate.

This environment simulates that process, allowing an agent to learn how to guide, not just evaluate.

🧩 Environment Design

The environment follows the OpenEnv standard:

reset() → starts a new session
step(action) → updates state based on action
state() → returns current state
Observation includes:
question, difficulty
answer score (0–10)
confidence
fatigue
Actions:
ask_easy_question
ask_medium_question
ask_hard_question
give_hint
give_feedback
change_topic
🏆 Reward & Behavior
Rewards improvement in candidate performance
Penalizes repetition and poor strategy
Models fatigue and learning effects
Encourages balanced interview flow
🎯 Tasks
Easy → achieve good score quickly
Medium → maintain performance over time
Hard → handle difficulty with consistency

Each task is graded using deterministic functions returning 0.0–1.0 scores.

▶️ Run
docker build -t interview-coach .
docker run -e HF_TOKEN="your_token" interview-coach
📤 Output Format
[START]
[STEP]
[END]
🐳 Deployment

Deployable on Hugging Face Spaces with Docker support.

🏁 Summary

This environment trains agents to think like interviewers, focusing on strategy, adaptation, and human learning dynamics.

💥 Training AI to become better interviewers.