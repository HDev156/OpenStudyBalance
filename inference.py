import os
from typing import List, Optional

from openai import OpenAI

from env import OpenStudyBalanceEnv
from models import Action
from graders import grade_final_state

# =========================================================
# REQUIRED ENVIRONMENT VARIABLES
# =========================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional

if not OPENAI_API_KEY:
    raise ValueError("Missing required environment variable: OPENAI_API_KEY or HF_TOKEN")

# =========================================================
# CONFIG
# =========================================================

TASK_NAME = "balanced_assignment_week"
BENCHMARK = "OpenStudyBalance"
MAX_STEPS = 20
MAX_TOTAL_REWARD = 10.0   # adjust if your env has a better known scale
SUCCESS_SCORE_THRESHOLD = 0.70

# =========================================================
# OPENAI CLIENT
# =========================================================

client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)

# =========================================================
# LOGGING HELPERS
# =========================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={done} error={error}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}",
        flush=True
    )

# =========================================================
# MODEL HELPER
# =========================================================

def get_model_action(observation) -> str:
    pending_tasks_str = "\n".join([
        f"- {t.name}: needs {t.hours_needed}h, priority {t.priority}, deadline in {t.deadline_days} days"
        for t in observation.pending_tasks
    ])

    prompt = f"""You are an AI student planner helping manage workload.

Your job is to choose EXACTLY ONE next action for the student.

Current State:
Pending Tasks:
{pending_tasks_str}

Available hours per day: {observation.available_hours_per_day}
Current stress level: {observation.stress_level:.2f}
Sleep hours: {observation.sleep_hours}
Message: {observation.message}

Available Actions (choose exactly one):
1. schedule_task <task_name> <day> <hours>
2. add_break <day>
3. finalize_plan

Rules:
- Prioritize urgent and high-priority tasks
- Avoid overloading a single day
- If stress is high, consider adding a break
- Respond with ONLY the action command
- No explanation, no extra text
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "finalize_plan"

# =========================================================
# ACTION PARSER
# =========================================================

def parse_action(action_text: str) -> Action:
    try:
        parts = action_text.strip().split()

        if not parts:
            return Action(action_type="finalize_plan")

        action_type = parts[0]

        if action_type == "schedule_task" and len(parts) >= 4:
            task_name = " ".join(parts[1:-2])
            day = int(parts[-2])
            hours = float(parts[-1])
            return Action(
                action_type="schedule_task",
                task_name=task_name,
                day=day,
                hours=hours
            )

        elif action_type == "add_break" and len(parts) == 2:
            day = int(parts[1])
            return Action(action_type="add_break", day=day)

        elif action_type == "finalize_plan":
            return Action(action_type="finalize_plan")

    except Exception:
        pass

    return Action(action_type="finalize_plan")

# =========================================================
# MAIN
# =========================================================

def main() -> None:
    env = OpenStudyBalanceEnv()

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        env.reset(TASK_NAME)
        observation = env.get_observation()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if env.done:
                break

            action_text = get_model_action(observation)
            action = parse_action(action_text)

            try:
                observation, reward, done, info = env.step(action)
                reward_value = reward.value if reward else 0.0
                error = None
            except Exception as exc:
                observation = env.get_observation()
                reward_value = 0.0
                done = True
                error = str(exc)

            rewards.append(reward_value)
            steps_taken = step
            last_reward = reward_value

            log_step(
                step=step,
                action=action_text,
                reward=reward_value,
                done=done,
                error=error
            )

            history.append(f"Step {step}: {action_text!r} -> reward {reward_value:+.2f}")

            if done:
                break

        # Final score
        try:
            final_score = grade_final_state(env.state())
            if isinstance(final_score, (int, float)):
                score = float(final_score)
            else:
                score = float(getattr(final_score, "value", 0.0))
        except Exception:
            score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0

        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()