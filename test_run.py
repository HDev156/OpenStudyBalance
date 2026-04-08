from env import OpenStudyBalanceEnv
from models import Action

# Create environment
env = OpenStudyBalanceEnv()

# Reset with easy task
env.reset("balanced_assignment_week")

# Print initial observation
initial_obs = env.state()
print("Initial State:")
print(f"  Tasks: {[t.name for t in initial_obs.tasks]}")
print(f"  Available hours/day: {initial_obs.available_hours_per_day}")
print(f"  Stress level: {initial_obs.stress_level}")
print(f"  Sleep hours: {initial_obs.sleep_hours}")
print()

# Take example actions
actions = [
    Action(action_type="schedule_task", task_name="Math Assignment", day=1, hours=2.0),
    Action(action_type="add_break", day=2),
    Action(action_type="finalize_plan")
]

for i, action in enumerate(actions, 1):
    print(f"Step {i}: {action.action_type}")
    if action.task_name:
        print(f"  Task: {action.task_name}, Day: {action.day}, Hours: {action.hours}")
    elif action.day:
        print(f"  Day: {action.day}")

    obs, reward, done, info = env.step(action)
    print(f"  Reward: {reward.value:.2f} ({reward.reason})")
    print(f"  Pending tasks: {len(obs.pending_tasks)}")
    print(f"  Stress level: {obs.stress_level:.2f}")
    print(f"  Done: {done}")
    print()

# Print final state
final_state = env.state()
print("Final State:")
print(f"  Tasks scheduled hours: {[(t.name, t.scheduled_hours) for t in final_state.tasks]}")
print(f"  Daily scheduled: {final_state.daily_scheduled}")
print(f"  Stress level: {final_state.stress_level}")
print(f"  Steps taken: {info['steps']}")