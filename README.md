# OpenStudyBalance

OpenStudyBalance is an OpenEnv-compatible environment for burnout-aware academic planning and student workload optimization. It simulates a student managing assignments, deadlines, study hours, stress, and sleep while trying to build a balanced study plan.

## Motivation

Students often face tight deadlines, busy weeks, and rising stress during exam season. OpenStudyBalance is designed as a hackathon-ready environment that makes this challenge into a simple planning and reinforcement learning problem. The goal is to encourage safe scheduling, avoid overload, and keep stress under control while still finishing important tasks.

## Environment Description

OpenStudyBalance models a student workload planner with:

- a set of academic tasks with deadlines, priorities, and required hours
- daily available study hours
- stress and sleep variables
- a simple action interface for scheduling and planning

The environment supports three tasks with increasing difficulty, and a simple reward/grade system to evaluate the final plan.

## Action Space

The environment supports three action types:

- `schedule_task`: assign study hours for a specific task on a chosen day
  - `task_name`: name of the task
  - `day`: integer day (1-7)
  - `hours`: number of hours to schedule
- `add_break`: add a break on a day to reduce stress
  - `day`: integer day (1-7)
- `finalize_plan`: end the planning episode and get a final score

Invalid or overloaded scheduling actions are penalized.

## Observation Space

Each observation contains:

- `pending_tasks`: list of pending `TaskItem` objects
- `available_hours_per_day`: float for daily study capacity
- `stress_level`: float from 0.0 to 1.0
- `sleep_hours`: current sleep estimate
- `message`: a short status message

A `TaskItem` includes:

- `name`
- `hours_needed`
- `deadline_days`
- `priority`
- `scheduled_hours`

## Tasks and Difficulty Levels

OpenStudyBalance includes three predefined scenarios:

1. **Balanced Assignment Week** (Easy)
   - A normal week with three assignments and reasonable deadlines.

2. **Deadline Collision Week** (Medium)
   - Several tasks due at the same time, forcing prioritization and careful scheduling.

3. **Burnout Prevention During Mid-Sems** (Hard)
   - A mid-semester exam period with many urgent tasks, low sleep, and high stress.

## Reward Design

The reward system is designed to encourage:

- scheduling high-priority tasks first
- respecting daily hour limits
- avoiding task overload
- balancing workload across days
- using breaks to manage stress

Actions that fail validation or overload a day receive negative rewards, while valid planning actions receive positive feedback.

## Grader Design

The final grading function scores the environment state between `0.0` and `1.0` using five factors:

- high-priority tasks scheduled
- deadlines respected
- workload overload avoided
- schedule balance
- stress management

Each factor contributes equally to the final score for a simple, deterministic evaluation.

## Setup Instructions

1. Create and activate a Python environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure your environment has the required files:

- `models.py`
- `tasks.py`
- `env.py`
- `graders.py`
- `inference.py`
- `test_run.py`
- `requirements.txt`
- `openenv.yaml`

## Running `inference.py`

Before running, set these environment variables:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="your_api_key_here"
```

Run the inference script:

```bash
python3 inference.py
```

The script will:

- reset the environment with the easy task
- prompt the model for action suggestions
- parse actions safely
- step through the environment
- print structured logs using `[START]`, `[STEP]`, and `[END]`
- compute a final score

## Why This Environment Is Useful

OpenStudyBalance is useful for hackathon and reinforcement learning experiments because it provides a realistic, easy-to-understand planning task with human-centered metrics like stress and sleep. It is also a strong candidate for model-based action selection, prompt engineering, and policy testing in a student-focused domain.

## Deployment to Hugging Face Spaces

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space
2. Choose "Docker" as the SDK
3. Set the Space name to `openstudybalance` (or your preferred name)
4. Upload all project files to the Space repository
5. The Space will automatically build and deploy using the provided Dockerfile
6. Your Space URL will be: `https://[username]-openstudybalance.hf.space`

The API will be available at:
- Health check: `GET /health`
- Reset environment: `POST /reset` with `{"task": "balanced_assignment_week"}`
- Step environment: `POST /step` with `{"action": {...}}`

## OpenEnv Submission

Once deployed, submit your environment to OpenEnv using the `openenv.yaml` configuration file. The space URL in the YAML should match your deployed Hugging Face Space.