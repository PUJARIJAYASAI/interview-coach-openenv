import random

class InterviewEnv:
    def __init__(self):
        self.question_bank = {
            "easy": [
                {"question": "Explain the difference between List and Set in Java.", "topic": "Java"},
                {"question": "What is a primary key in a database?", "topic": "SQL"},
                {"question": "Reverse a string in Python without using [::-1].", "topic": "DSA"},
                {"question": "Tell me about yourself.", "topic": "HR"}
            ],
            "medium": [
                {"question": "Describe the concept of polymorphism with an example.", "topic": "Java"},
                {"question": "What are the common time and space complexities of Merge Sort?", "topic": "DSA"},
                {"question": "Explain the difference between checked and unchecked exceptions.", "topic": "Java"},
                {"question": "Why should we hire you?", "topic": "HR"}
            ],
            "hard": [
                {"question": "How does the garbage collector work in Java? Mention different stages.", "topic": "Java"},
                {"question": "Explain how a B-Tree differs from a B+ Tree.", "topic": "DSA"},
                {"question": "Design a system that tracks the real-time leaderboard for a gaming platform.", "topic": "System Design"},
                {"question": "How do you handle conflict in a team?", "topic": "HR"}
            ]
        }
        self.learning_factor = 0.3
        self.fatigue = 0
        self.reset()

    def reset(self, task_name="easy"):
        self.task_name = task_name
        self.current_topic = random.choice(["Java", "DSA", "HR", "SQL"])
        self.current_difficulty = "easy"
        self.current_question = self._pick_question(self.current_difficulty)
        self.candidate_answer = "I'm ready to begin."
        self.answer_score = 5
        self.history = []
        self.last_actions = []
        self.learning_factor = 0.3
        self.fatigue = 0
        self.steps = 0
        step_limits = {"easy": 5, "medium": 10, "hard": 15}
        self.max_steps = step_limits.get(task_name, 10)
        return self.state()

    def _pick_question(self, difficulty):
        choices = self.question_bank[difficulty]
        return random.choice(choices)["question"]

    def get_valid_actions(self):
        if not self.last_actions:
            return ["ask_easy_question", "ask_medium_question"]

        return [
            "ask_easy_question",
            "ask_medium_question",
            "ask_hard_question",
            "give_hint",
            "give_feedback"
        ]

    def simulate_answer(self, difficulty):
        base = 5 + int(self.learning_factor * 5) - self.fatigue
        if difficulty == "easy":
            base += 1
        elif difficulty == "hard":
            base -= 2
        return max(0, min(10, int(base)))

    def state(self):
        confidence = max(0, min(1, self.learning_factor - (self.fatigue * 0.1)))
        return {
            "task_name": self.task_name,
            "question": self.current_question,
            "difficulty": self.current_difficulty,
            "candidate_answer": self.candidate_answer,
            "answer_score": self.answer_score,
            "history": self.history[-5:],
            "learning_factor": round(self.learning_factor, 2),
            "fatigue": round(self.fatigue, 2),
            "confidence": round(confidence, 2)
        }

    def step(self, action):
        self.steps += 1
        old_score = self.answer_score

        if "ask" in action:
            self.fatigue += 0.2
        else:
            self.fatigue = max(0, self.fatigue - 0.1)

        if action in ["give_hint", "give_feedback"]:
            self.learning_factor += 0.05

        if action == "give_hint":
            self.answer_score = self.simulate_answer(self.current_difficulty) + 1
            self.candidate_answer = "Thanks for the hint! I'll try to incorporate that."

        elif action == "give_feedback":
            feedback_count = self.last_actions.count("give_feedback")
            if feedback_count == 0:
                self.answer_score = self.simulate_answer(self.current_difficulty) + 2
            elif feedback_count == 1:
                self.answer_score = self.simulate_answer(self.current_difficulty) + 1
            else:
                self.answer_score = self.simulate_answer(self.current_difficulty)
            self.candidate_answer = "I appreciate the feedback. Let me improve."

        elif action == "ask_easy_question":
            self.current_difficulty = "easy"
            self.current_question = self._pick_question("easy")
            self.answer_score = self.simulate_answer("easy")

        elif action == "ask_medium_question":
            self.current_difficulty = "medium"
            self.current_question = self._pick_question("medium")
            self.answer_score = self.simulate_answer("medium")

        elif action == "ask_hard_question":
            self.current_difficulty = "hard"
            self.current_question = self._pick_question("hard")
            self.answer_score = self.simulate_answer("hard")

        elif action == "change_topic":
            self.current_topic = random.choice(["Java", "DSA", "HR", "SQL"])
            self.current_question = self._pick_question(self.current_difficulty)

        if not self.last_actions and action in ["give_feedback", "give_hint"]:
            return self.state(), -5.0, False, {"error": "invalid_start_action"}

        new_score = self.answer_score
        if new_score - old_score > 2:
            new_score = old_score + 2
        elif old_score - new_score > 2:
            new_score = old_score - 2
        self.answer_score = new_score

        reward = (new_score - old_score) * 1.2
        reward += random.uniform(-0.7, 0.7)

        question_actions = ["ask_easy_question", "ask_medium_question", "ask_hard_question"]
        if not self.last_actions and action in question_actions:
            reward += 2

        if action == "give_feedback" and self.last_actions.count("give_feedback") >= 2:
            reward -= 5

        if self.last_actions and action in ["give_hint", "give_feedback"]:
            if not any(a in question_actions for a in self.last_actions):
                reward -= 4

        if action in self.last_actions[-2:]:
            reward -= 3

        if action not in self.last_actions[-3:]:
            reward += 1

        if action == "ask_hard_question" and old_score < 4:
            reward -= 3

        if new_score >= 8:
            reward += 10

        # Normalize reward from [-4.0, 4.0] to [0.01, 0.99]
        reward = (reward + 4) / 8
        reward = max(0.01, min(0.99, reward))

        self.last_actions.append(action)
        self.history.append({"action": action, "score": self.answer_score})
        done = self.steps >= self.max_steps
        
        return self.state(), float(reward), done, {}

    def close(self):
        pass
