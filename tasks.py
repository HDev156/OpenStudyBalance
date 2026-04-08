from models import TaskItem, EnvironmentState


def get_balanced_assignment_week() -> EnvironmentState:
    """
    Easy scenario: Balanced Assignment Week
    A typical week with 3 assignments spread out over the week.
    """
    tasks = [
        TaskItem(name="Math Assignment", hours_needed=4.0, deadline_days=5, priority=2, scheduled_hours=0.0),
        TaskItem(name="Physics Lab Report", hours_needed=3.0, deadline_days=3, priority=3, scheduled_hours=0.0),
        TaskItem(name="Programming Project", hours_needed=6.0, deadline_days=7, priority=1, scheduled_hours=0.0),
    ]
    return EnvironmentState(
        current_day=0,
        tasks=tasks,
        available_hours_per_day=8.0,
        stress_level=0.2,
        sleep_hours=8.0
    )


def get_deadline_collision_week() -> EnvironmentState:
    """
    Medium scenario: Deadline Collision Week
    Multiple assignments due on the same day, creating time pressure.
    """
    tasks = [
        TaskItem(name="Data Structures Quiz", hours_needed=2.0, deadline_days=2, priority=3, scheduled_hours=0.0),
        TaskItem(name="Chemistry Experiment", hours_needed=5.0, deadline_days=2, priority=2, scheduled_hours=0.0),
        TaskItem(name="English Essay", hours_needed=4.0, deadline_days=2, priority=1, scheduled_hours=0.0),
        TaskItem(name="Electronics Homework", hours_needed=3.0, deadline_days=5, priority=2, scheduled_hours=0.0),
    ]
    return EnvironmentState(
        current_day=0,
        tasks=tasks,
        available_hours_per_day=6.0,
        stress_level=0.5,
        sleep_hours=7.0
    )


def get_burnout_prevention_mid_sems() -> EnvironmentState:
    """
    Hard scenario: Burnout Prevention During Mid-Sems
    High workload during mid-semester exams with low sleep and high stress.
    """
    tasks = [
        TaskItem(name="Mid-Sem Math Exam Prep", hours_needed=8.0, deadline_days=3, priority=1, scheduled_hours=0.0),
        TaskItem(name="Physics Mid-Term Study", hours_needed=7.0, deadline_days=3, priority=1, scheduled_hours=0.0),
        TaskItem(name="Programming Final Project", hours_needed=10.0, deadline_days=5, priority=2, scheduled_hours=0.0),
        TaskItem(name="Chemistry Lab Analysis", hours_needed=4.0, deadline_days=2, priority=3, scheduled_hours=0.0),
        TaskItem(name="Group Presentation Prep", hours_needed=6.0, deadline_days=4, priority=2, scheduled_hours=0.0),
    ]
    return EnvironmentState(
        current_day=0,
        tasks=tasks,
        available_hours_per_day=5.0,
        stress_level=0.8,
        sleep_hours=6.0
    )


def get_task_by_name(task_name: str) -> EnvironmentState:
    """
    Helper function to get a predefined task by name.
    """
    task_map = {
        "balanced_assignment_week": get_balanced_assignment_week,
        "deadline_collision_week": get_deadline_collision_week,
        "burnout_prevention_mid_sems": get_burnout_prevention_mid_sems,
    }
    if task_name not in task_map:
        raise ValueError(f"Unknown task name: {task_name}")
    return task_map[task_name]()