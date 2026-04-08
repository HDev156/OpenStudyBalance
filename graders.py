from models import EnvironmentState, TaskItem
from typing import List, Dict


def grade_final_state(state: EnvironmentState) -> float:
    """
    Compute multi-objective final evaluation and return overall score.
    """
    scores = compute_detailed_scores(state)
    return scores["overall"]


def compute_detailed_scores(state: EnvironmentState) -> Dict[str, float]:
    """
    Compute detailed sub-scores for multi-objective evaluation.
    Returns dict with all scores including overall.
    """
    # Deadline Adherence Score
    total_tasks = len(state.tasks)
    if total_tasks == 0:
        deadline_score = 1.0
    else:
        completed_on_time = sum(1 for t in state.tasks if t.scheduled_hours >= t.hours_needed or t.deadline_days > state.current_day)
        deadline_score = completed_on_time / total_tasks

    # Workload Balance Score
    daily_hours = [state.daily_scheduled.get(day, 0) for day in range(1, 8)]
    if daily_hours:
        avg_hours = sum(daily_hours) / len(daily_hours)
        variance = sum((h - avg_hours) ** 2 for h in daily_hours) / len(daily_hours)
        std_dev = variance ** 0.5
        # Lower std dev = better balance, cap at available_hours_per_day
        max_std = state.available_hours_per_day
        balance_score = max(0.0, 1.0 - (std_dev / max_std))
    else:
        balance_score = 1.0

    # Burnout Safety Score
    burnout_score = 1.0 - state.burnout_risk  # Lower risk = higher score

    # Priority Handling Score
    high_priority_tasks = [t for t in state.tasks if t.priority <= 2]
    if high_priority_tasks:
        high_priority_completed = sum(1 for t in high_priority_tasks if t.scheduled_hours > 0)
        priority_score = high_priority_completed / len(high_priority_tasks)
    else:
        priority_score = 1.0

    # Overall Score (weighted average)
    overall = (deadline_score * 0.3 + balance_score * 0.2 + burnout_score * 0.25 + priority_score * 0.25)

    return {
        "deadline_adherence": deadline_score,
        "workload_balance": balance_score,
        "burnout_safety": burnout_score,
        "priority_handling": priority_score,
        "overall": overall
    }


def get_performance_summary(scores: Dict[str, float]) -> str:
    """Generate a natural language summary of the performance."""
    overall = scores["overall"]
    deadline = scores["deadline_adherence"]
    balance = scores["workload_balance"]
    burnout = scores["burnout_safety"]
    priority = scores["priority_handling"]
    
    if overall >= 0.85:
        summary = "Outstanding planning! The schedule excellently balanced all objectives."
    elif overall >= 0.7:
        summary = "Good work overall. The plan handled most challenges well."
    elif overall >= 0.5:
        summary = "Fair performance. Some areas need improvement."
    else:
        summary = "The plan needs significant improvement in multiple areas."
    
    issues = []
    if deadline < 0.7:
        issues.append("deadline adherence")
    if balance < 0.7:
        issues.append("workload balance")
    if burnout < 0.7:
        issues.append("burnout management")
    if priority < 0.7:
        issues.append("priority handling")
    
    if issues:
        summary += f" Areas needing attention: {', '.join(issues)}."
    
    return summary


# Helper functions if needed
def is_high_priority(task: TaskItem) -> bool:
    return task.priority <= 2


def is_deadline_respected(task: TaskItem, current_day: int) -> bool:
    return task.scheduled_hours >= task.hours_needed or task.deadline_days > current_day