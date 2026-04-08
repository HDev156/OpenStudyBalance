from models import EnvironmentState, TaskItem, Action, Observation, Reward
from tasks import get_task_by_name
from graders import grade_final_state
import random


class OpenStudyBalanceEnv:
    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self._state = None
        self.done = False
        self.steps = 0
        self.last_event = None  # Track the last unexpected event

    def reset(self, task_name: str) -> Observation:
        self._state = get_task_by_name(task_name)
        self.done = False
        self.steps = 0
        self.last_event = None
        self._normalize_daily_scheduled()
        self._update_burnout_risk()  # Initialize burnout risk
        return self.get_observation()

    def state(self) -> EnvironmentState:
        return self._state

    def get_observation(self) -> Observation:
        self._normalize_daily_scheduled()
        return self._get_observation()

    def _get_observation(self) -> Observation:
        pending = [t for t in self._state.tasks if t.scheduled_hours < t.hours_needed]
        message = f"Day {self._state.current_day}, {len(pending)} pending tasks"
        return Observation(
            pending_tasks=pending,
            available_hours_per_day=self._state.available_hours_per_day,
            stress_level=self._state.stress_level,
            sleep_hours=self._state.sleep_hours,
            message=message
        )

    def _normalize_daily_scheduled(self):
        """Ensure daily_scheduled keys are integers for safe access."""
        if self._state is None:
            return
        self._state.daily_scheduled = {
            int(day): float(hours)
            for day, hours in self._state.daily_scheduled.items()
        }

    def _update_burnout_risk(self):
        """Compute burnout risk based on current state factors."""
        self._normalize_daily_scheduled()
        stress_factor = self._state.stress_level
        sleep_factor = max(0, (8.0 - self._state.sleep_hours) / 8.0)  # Lower sleep = higher risk
        urgent_tasks = sum(1 for t in self._state.tasks if t.priority <= 2 and t.deadline_days <= 3)
        urgent_factor = min(urgent_tasks / 3.0, 1.0)  # Cap at 3 urgent tasks
        overload_factor = sum(1 for hours in self._state.daily_scheduled.values() if hours > self._state.available_hours_per_day) / 7.0
        
        # Weighted combination
        self._state.burnout_risk = min(1.0, (stress_factor * 0.3 + sleep_factor * 0.2 + urgent_factor * 0.3 + overload_factor * 0.2))

    def get_burnout_category(self) -> str:
        """Return human-readable burnout risk category."""
        risk = self._state.burnout_risk
        if risk < 0.25:
            return "Low"
        elif risk < 0.5:
            return "Moderate"
        elif risk < 0.75:
            return "High"
        else:
            return "Critical"

    def inject_disruption(self):
        """Manually trigger an unexpected event."""
        self._apply_random_event()

    def _apply_random_event(self):
        """Apply a random unexpected event to the current state."""
        events = [
            {
                "name": "Surprise Quiz Announced",
                "description": "A surprise quiz was announced for tomorrow, adding 2 hours of study time.",
                "effect": lambda state: state.tasks.append(TaskItem(
                    name="Surprise Quiz Prep", hours_needed=2.0, deadline_days=1, priority=1, scheduled_hours=0.0
                ))
            },
            {
                "name": "Lab Viva Moved Earlier",
                "description": "Your lab viva was moved to tomorrow, increasing urgency.",
                "effect": lambda state: [t.__setattr__('deadline_days', max(1, t.deadline_days - 1)) for t in state.tasks if 'Lab' in t.name]
            },
            {
                "name": "Group Meeting Added",
                "description": "An urgent group meeting was scheduled for day 2, reducing available hours.",
                "effect": lambda state: state.daily_scheduled.__setitem__(2, state.daily_scheduled.get(2, 0) + 2.0)
            },
            {
                "name": "Poor Sleep Night",
                "description": "You had a poor sleep night, reducing sleep hours by 2.",
                "effect": lambda state: setattr(state, 'sleep_hours', max(4.0, state.sleep_hours - 2.0))
            },
            {
                "name": "Power Cut",
                "description": "A power cut reduced study time on day 3 by 2 hours.",
                "effect": lambda state: state.daily_scheduled.__setitem__(3, max(0, state.daily_scheduled.get(3, 0) - 2.0))
            },
            {
                "name": "Extra Assignment",
                "description": "An extra assignment was added with 3 hours needed in 4 days.",
                "effect": lambda state: state.tasks.append(TaskItem(
                    name="Extra Assignment", hours_needed=3.0, deadline_days=4, priority=2, scheduled_hours=0.0
                ))
            }
        ]
        
        event = random.choice(events)
        event["effect"](self._state)
        self.last_event = event
        self._update_burnout_risk()  # Recalculate after event

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done:
            return self._get_observation(), Reward(value=0.0, reason="Episode already done"), True, {}

        self.steps += 1
        reward_value = 0.0
        reason = ""

        # Auto-trigger event every 5 steps
        if self.steps % 5 == 0 and self.steps > 0:
            self._apply_random_event()

        # Apply action
        if action.action_type == "schedule_task":
            if not action.task_name or action.day is None or action.hours is None:
                reward_value = -0.1
                reason = "Invalid schedule_task action: missing parameters"
            else:
                task = next((t for t in self._state.tasks if t.name == action.task_name), None)
                if not task:
                    reward_value = -0.1
                    reason = f"Task {action.task_name} not found"
                elif action.hours <= 0 or action.hours > self._state.available_hours_per_day:
                    reward_value = -0.1
                    reason = "Invalid hours for scheduling"
                else:
                    if action.day < 1 or action.day > 7:
                        reward_value = -0.1
                        reason = "Invalid day"
                    else:
                        current_scheduled = self._state.daily_scheduled.get(action.day, 0.0)
                        if current_scheduled + action.hours > self._state.available_hours_per_day:
                            reward_value = -0.2
                            reason = "Overload on day"
                            self._state.stress_level = min(1.0, self._state.stress_level + 0.1)
                        else:
                            self._state.daily_scheduled[action.day] = current_scheduled + action.hours
                            task.scheduled_hours += action.hours
                            reward_value = 0.1 if task.priority <= 2 else 0.05
                            reason = f"Scheduled {action.hours}h for {task.name} on day {action.day}"
                            # Reduce burnout risk slightly for good planning
                            self._state.burnout_risk = max(0.0, self._state.burnout_risk - 0.02)
        elif action.action_type == "add_break":
            if action.day is None:
                reward_value = -0.1
                reason = "Invalid add_break: missing day"
            else:
                self._state.stress_level = max(0.0, self._state.stress_level - 0.05)
                self._state.burnout_risk = max(0.0, self._state.burnout_risk - 0.03)
                reward_value = 0.05
                reason = f"Added break on day {action.day}"
        elif action.action_type == "finalize_plan":
            self.done = True
            final_score = grade_final_state(self._state)
            reward_value = final_score
            reason = f"Plan finalized with score {final_score}"
        else:
            reward_value = -0.1
            reason = f"Unknown action type: {action.action_type}"

        # Update burnout risk after action
        self._update_burnout_risk()

        # Check max steps
        if self.steps >= self.max_steps:
            self.done = True
            if reason == "":
                final_score = grade_final_state(self._state)
                reward_value += final_score * 0.5
                reason += f" Max steps reached, partial score {final_score}"

        obs = self._get_observation()
        reward = Reward(value=reward_value, reason=reason)
        info = {"steps": self.steps}
        return obs, reward, self.done, info